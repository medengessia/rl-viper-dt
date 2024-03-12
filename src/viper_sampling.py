import os
import time
import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from joblib import dump
from sklearn import tree
from stable_baselines3 import DQN


def get_policy_nn(env, algo_rl, n_iter, path_to_expert):
    """Generates a RL policy by training a RL Neural Network and returns the policy.

    Args:
        env (gym.Env): The environment in which the models learn.
        algo_rl : A RL Algorithm.
        n_iter (int): Number of iterations.
        path_to_expert (str): a path to an existing policy.

    Returns:
        The obtained policy.
    """
    model = algo_rl('MlpPolicy', env, verbose=1)

    if path_to_expert is not None:
        model = algo_rl.load(path_to_expert, optimize_memory_usage=False)
    else:
        model.learn(n_iter) # Training the policy

    return model


def generate_data(env, model, n_iter):
    """Generates data, with actions weighted by their reward.

    Args:
        env (gym.Env): The environment in which the models learn.
        model : A DRL model.
        n_iter (int): Number of iterations.

    Returns:
        Ndarray: The obtained dataset.
    """
    s, _ = env.reset()
    S, A = [], []

    for _ in range(n_iter):
        S.append(s)   # Current state

        action, _ = model.predict(s, deterministic=True) # Entering a state
        A.append(action) # Chosen action

        new_s, _, terminated, truncated, _ = env.step(action) # Taking action
        s = new_s # Getting a new state

        if terminated or truncated:
            s, _ = env.reset()

    return np.array(S), np.array(A)


def fit_dtree(S, A, Is, algo_dt, depth=None, leaf_nodes=None, reg=None):
    """Fits a decision tree to the policy-generated dataset and returns it with its score.

    Args:
        S (Ndarray): The sampled State space.
        A (Ndarray): The sampled Action space.
        Is (Ndarray): The sampled Importance for each couple (S,A).
        algo_dt : A decision tree algorithm.
        depth (int): The tree desired maximal depth. Defaults to None.
        leaf_nodes (int): The tree desired maximal leaf nodes. Defaults to None.
        reg (str): The regularization mode. Defaults to None.

    Returns:
        list: A list of trained decicision trees.
    """
    if reg == 'max_depth':
      dt = algo_dt(max_depth=depth)

    elif reg == 'max_leaf_nodes':
      dt = algo_dt(max_leaf_nodes=leaf_nodes)

    else:
      dt = algo_dt()

    dt.fit(S, A, Is)
    acc = dt.score(S, A, Is)

    print("Score of fitted DT is {}".format(acc))

    return dt, acc


def eval(dt, env, n_iter=5000):
    """Evaluates a decision tree by giving its mean cumulative reward.

    Args:
        dt (tree): The decision tree to evaluate.
        env (gym.Env): The environment in which the models learn.
        n_iter (int, optional): Number of iterations. Defaults to 10.

    Returns:
        float: The mean cumulative reward of a tree over the iterations.
    """
    S = []
    s, _ = env.reset()
    tot_steps = 0
    rew_per_trajs = []
    tot_rew_current_traj = 0

    while tot_steps < n_iter:
        done = False
        S.append(s)
        action = dt.predict([s]) # Entering a state
        new_s, reward, terminated, truncated, _ = env.step(int(action[0])) # The decision tree imitates a RL agent
        tot_rew_current_traj += reward
        tot_steps += 1
        done = terminated or truncated

        if done:
            s, _ = env.reset()
            rew_per_trajs.append(tot_rew_current_traj)
            tot_rew_current_traj = 0
        else:
            s = new_s # Getting a new state

    eval_dt = np.mean(rew_per_trajs)
    print("DT eval: {}".format(eval_dt))

    return np.array(S), eval_dt


def get_perf_expert(env, model, n_iter=10):
    """Evaluates an expert RL policy by giving its mean cumulative reward.

    Args:
        env (gym.Env): The environment in which the models learn.
        model : A DRL model.
        n_iter (int, optional): Number of iterations. Defaults to 10.

    Returns:
        float: The mean cumulative reward of an expert RL policy over the iterations.
    """
    avg = 0
    for _ in range(n_iter):
        s, _ = env.reset()
        done = False
        tot = 0
        while not done:
            action = model.predict(s, deterministic=True)[0]
            s, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            tot += r
        avg += tot
    return avg / n_iter


def viper(env, model, algo_dt, iter_viper, nb_data_from_nnpolicy, depth=None, leaf_nodes=None, reg=None):
    """Proposes an implementation of Viper algorithm.

    Args:
        env (gym.Env): The environment in which the models learn.
        model : A DRL model.
        algo_dt : A decision tree algorithm.
        iter_viper (int): Number of iterations for VIPER.
        nb_data_from_nnpolicy (int): Number of iterations for data generating functions.
        depth (int): The tree desired maximal depth. Defaults to None.
        leaf_nodes (int): The tree desired maximal leaf nodes. Defaults to None.
        reg (str): The regularization mode. Defaults to None.

    Returns:
        Ndarray: The mean cumulative rewards of the trees.
    """
    best_dt_eval = -np.inf
    best_dt = None    

    list_acc, list_eval = [], []

    for i in range(iter_viper):
        print("#### ITER {} ####".format(i+1))

        if i == 0:
            S, A = generate_data(env, model, nb_data_from_nnpolicy)
            DS = S
            DA = A

            with torch.no_grad():
                qs_a = model.q_net(model.q_net.obs_to_tensor(S)[0])

            Is = torch.abs(qs_a.amax(axis=-1) - qs_a.amin(axis=-1)).numpy()

        if reg == 'max_depth':
            dt, acc = fit_dtree(DS, DA, Is, algo_dt, depth=depth, reg=reg)

        elif reg == 'max_leaf_nodes':
            dt, acc = fit_dtree(DS, DA, Is, algo_dt, leaf_nodes=leaf_nodes, reg=reg)

        else:
            dt, acc = fit_dtree(DS, DA, Is, algo_dt)
        
        Sdt, eval_dt = eval(dt, env)

        if eval_dt > best_dt_eval:
            best_dt_eval = eval_dt
            best_dt = dt

        list_acc.append(acc)
        list_eval.append(eval_dt)
        DS = np.concatenate((DS, Sdt))

        with torch.no_grad():
            qs_a = model.q_net(model.q_net.obs_to_tensor(Sdt)[0])

        DA = np.concatenate((DA, qs_a.argmax(axis=1).numpy()))
        Is = np.concatenate((Is, torch.abs(qs_a.amax(axis=-1) - qs_a.amin(axis=-1)).numpy()))

    return best_dt, best_dt_eval, np.array(list_acc), np.array(list_eval)


if __name__ == "__main__":

    algo_dt = tree.DecisionTreeClassifier
    algo_rl = DQN
    iter_viper = 20
    nb_data_from_nn_policy = 10_000

    for regul_type in ["max_depth", "max_leaf_nodes"]:

        for k, env_name in enumerate(["Acrobot-v1", "CartPole-v1", "LunarLander-v2", "MountainCar-v0"]):

            env = gym.make(env_name)
            path_to_expert = 'policies/' + env_name + '.zip'
            model = get_policy_nn(env, algo_rl, nb_data_from_nn_policy, path_to_expert)
            perf_expert = get_perf_expert(env, model)
            print(perf_expert)

            colors = ["black", "red", "blue", "green"]
                
            for i, d in enumerate([2,3,4,5]):
                mean_acc = 0
                mean_eval = 0

                if regul_type == "max_leaf_nodes":
                    l_nodes = 2**d - 1

                    for seed in range(5):
                        file_path = os.path.join("experiments_viper/{}/{}/{}/seed{}".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed))
                        os.makedirs(file_path, exist_ok=True)

                        # START TIMER 
                        start = time.time()

                        best_dt, best_dt_eval, list_acc, list_eval = viper(env, model, algo_dt, iter_viper, nb_data_from_nn_policy, leaf_nodes=l_nodes, reg=regul_type)

                        end = time.time()
                        # END TIMER

                        elapsed = end - start
                        np.save("experiments_viper/{}/{}/{}/seed{}/list_acc.npy".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed), np.hstack((list_acc, np.array([np.mean(list_acc), np.std(list_acc)]))))
                        np.save("experiments_viper/{}/{}/{}/seed{}/list_eval.npy".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed), np.hstack((list_eval, np.array([np.mean(list_eval), np.std(list_eval)]))))
                        np.save("experiments_viper/{}/{}/{}/seed{}/best_dt_eval.npy".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed), np.array([best_dt_eval]))
                        np.save("experiments_viper/{}/{}/{}/seed{}/algo_duration_in_seconds.npy".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed), np.array([elapsed]))
                        dump(best_dt, "experiments_viper/{}/{}/{}/seed{}/best_dt.joblib".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed))
                        
                        if env_name == "Acrobot-v1":
                            list_eval_norm = (list_eval+500)/(perf_expert+500)

                        elif env_name == "MountainCar-v0":
                            list_eval_norm = (list_eval+200)/(perf_expert+200)

                        elif env_name == "LunarLander-v2":
                            list_eval_norm = (list_eval+1000)/(perf_expert+1000)

                        else:
                            list_eval_norm = list_eval/perf_expert

                        mean_acc = mean_acc + list_acc
                        mean_eval = mean_eval + list_eval_norm

                    # plt.plot(mean_acc/5, label="mean-accuracy-leaf_nodes-{}".format(l_nodes), c=colors[i], linestyle = "dotted")
        
                    plt.plot(mean_eval/5, label="mean-eval-leaf_nodes-{}".format(l_nodes), c=colors[i])
                    plt.fill_between(np.arange(iter_viper), (mean_eval - np.std(mean_eval))/5, (mean_eval + np.std(mean_eval))/5, color=colors[i], alpha=0.3)

                else:

                    for seed in range(5):
                        file_path = os.path.join("experiments_viper/{}/{}/{}/seed{}".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed))
                        os.makedirs(file_path, exist_ok=True)

                        # START TIMER
                        start = time.time()

                        best_dt, best_dt_eval, list_acc, list_eval = viper(env, model, algo_dt, iter_viper, nb_data_from_nn_policy, depth=d, reg=regul_type)

                        end = time.time()
                        # END TIMER

                        elapsed = end - start
                        np.save("experiments_viper/{}/{}/{}/seed{}/list_acc.npy".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed), np.hstack((list_acc, np.array([np.mean(list_acc), np.std(list_acc)]))))
                        np.save("experiments_viper/{}/{}/{}/seed{}/list_eval.npy".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed), np.hstack((list_eval, np.array([np.mean(list_eval), np.std(list_eval)]))))
                        np.save("experiments_viper/{}/{}/{}/seed{}/best_dt_eval.npy".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed), np.array([best_dt_eval]))
                        np.save("experiments_viper/{}/{}/{}/seed{}/algo_duration_in_seconds.npy".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed), np.array([elapsed]))
                        dump(best_dt, "experiments_viper/{}/{}/{}/seed{}/best_dt.joblib".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed))
    
                        if env_name == "Acrobot-v1":
                            list_eval_norm = (list_eval+500)/(perf_expert+500)

                        elif env_name == "MountainCar-v0":
                            list_eval_norm = (list_eval+200)/(perf_expert+200)

                        elif env_name == "LunarLander-v2":
                            list_eval_norm = (list_eval+1000)/(perf_expert+1000)

                        else:
                            list_eval_norm = list_eval/perf_expert
                            
                        mean_acc = mean_acc + list_acc
                        mean_eval = mean_eval + list_eval_norm

                    # plt.plot(mean_acc/5, label="mean-accuracy-depth-{}".format(d), c=colors[i], linestyle = "dotted")

                    plt.plot(mean_eval/5, label="mean-eval-depth-{}".format(d), c=colors[i])
                    plt.fill_between(np.arange(iter_viper), (mean_eval - np.std(mean_eval))/5, (mean_eval + np.std(mean_eval))/5, color=colors[i], alpha=0.3)
                
            plt.legend()
            plt.title(env_name)
            plt.xlabel("Viper Iteration")
            plt.grid()
            plt.savefig("experiments_viper/{}/{}/res_{}_{}.pdf".format(regul_type, env_name, env_name, regul_type)) # 1 ENV + 1 regul type
            plt.clf()