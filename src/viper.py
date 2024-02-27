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
    policy = algo_rl('MlpPolicy', env, verbose=1)

    if path_to_expert is not None:
        policy = algo_rl.load(path_to_expert)
    else:
        policy.learn(n_iter) # Training the policy

    return policy


def generate_data(env, policy, n_iter):
    """Generates data, with actions weighted by their reward.

    Args:
        env (gym.Env): The environment in which the models learn.
        policy : An expert RL policy.
        n_iter (int): Number of iterations.

    Returns:
        Ndarray: The obtained dataset.
    """
    s, _ = env.reset()
    S, A = np.zeros((n_iter, env.observation_space.shape[0])), np.zeros((n_iter,1))

    for i in range(n_iter):
        S[i] = s   # Current state

        action, _ = policy.predict(s, deterministic=True) # Entering a state

        new_s, _, terminated, truncated, _ = env.step(action) # Taking action
        s = new_s # Getting a new state

        if terminated or truncated:
            s, _ = env.reset()

    return S, A


def fit_dtree(S, A, algo_dt, depth):
    """Fits a decision tree to the policy-generated dataset and returns it with its score.

    Args:
        S (Ndarray): The sampled State space.
        A (Ndarray): The sampled Action space.
        algo_dt : A decision tree algorithm.
        depth (int): The tree desired maximal depth.

    Returns:
        list: A list of trained decicision trees.
    """
    dt = algo_dt(max_depth=depth)
    dt.fit(S, A)
    acc = dt.score(S, A)

    print("Score of fitted DT is {}".format(acc))

    return dt, acc


def eval(dt, env, n_iter=10):
    """Evaluates a decision tree by giving its mean cumulative reward.

    Args:
        dt (tree): The decision tree to evaluate.
        env (gym.Env): The environment in which the models learn.
        n_iter (int, optional): Number of iterations. Defaults to 10.

    Returns:
        float: The mean cumulative reward of a tree over the iterations.
    """
    S = []
    score = 0

    for _ in range(n_iter):
        r_traj = 0
        done = False
        s, _ = env.reset()

        while not done:
            S.append(s)
            action = dt.predict([s]) # Entering a state
            new_s, reward, terminated, truncated, _ = env.step(int(action[0])) # The decision tree imitates a RL agent

            s = new_s # Getting a new state
            r_traj += reward
            done = terminated or truncated
        score += r_traj
    
    eval_dt = score / n_iter
    print("DT eval: {}".format(score))

    return np.array(S), eval_dt


def get_perf_expert(env, policy, n_iter=10):
    """Evaluates an expert RL policy by giving its mean cumulative reward.

    Args:
        env (gym.Env): The environment in which the models learn.
        policy : An expert RL policy.
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
            action = policy.predict(s)[0]
            s, r, term, trunc, _ = env.step(action)
            done = term or trunc
            tot += r
        avg += tot
    return avg / n_iter


def viper(env, policy, algo_dt, iter_viper, nb_data_from_nnpolicy, depth):
    """Proposes an implementation of Viper algorithm.

    Args:
        env (gym.Env): The environment in which the models learn.
        policy : An expert RL policy.
        algo_dt : A decision tree algorithm.
        iter_viper (int): Number of iterations for VIPER.
        nb_data_from_nnpolicy (int): Number of iterations for data generating functions.
        depth (int): The tree desired maximal depth.

    Returns:
        Ndarray: The mean cumulative rewards of the trees.
    """
    best_dt_eval = -np.inf
    best_dt = None
    list_acc, list_eval, list_dtrees = [], [], []

    for i in range(iter_viper):
        print("#### ITER {} ####".format(i+1))

        if i == 0:
            S, A = generate_data(env, policy, nb_data_from_nnpolicy)
            DS = S
            DA = A

        dt, acc = fit_dtree(DS, DA, algo_dt, depth)
        Sdt, eval_dt = eval(dt, env)

        if eval_dt > best_dt_eval:
            best_dt_eval = eval_dt
            best_dt = dt

        list_dtrees.append(dt)
        list_acc.append(acc)
        list_eval.append(eval_dt)

        A_Sdt = policy.predict(Sdt)[0]
        DS = np.concatenate((DS, Sdt))
        DA = np.concatenate((DA, A_Sdt.reshape(A_Sdt.shape[0], 1)))

    return best_dt, np.array(list_acc), np.array(list_eval)


if __name__ == "__main__":

    algo_dt = tree.DecisionTreeClassifier
    algo_rl = DQN
    iter_viper = 20
    nb_data_from_nn_policy = 10_000

    for k, env_name in enumerate(["LunarLander-v2"]):

        env = gym.make(env_name)
        path_to_expert = 'policies/' + env_name + '.zip'
        policy = get_policy_nn(env, algo_rl, nb_data_from_nn_policy, path_to_expert)
        perf_expert = get_perf_expert(env, policy)
        print(perf_expert)

        colors = ["black", "red", "blue"]
        for i, d in enumerate([4,6,8]):

            mean_acc = 0
            mean_eval = 0
            for seed in range(5):
                best_dt, list_acc, list_eval = viper(env, policy, algo_dt, iter_viper, nb_data_from_nn_policy, depth=d)
                dump(best_dt, "saved_dt_gymnasium/{}/depth{}/seed{}.joblib".format(env_name,d,seed))
                list_eval = list_eval/perf_expert
                mean_acc = mean_acc + list_acc
                mean_eval= mean_eval + list_eval

            plt.plot(mean_acc/1, label="mean-accuracy-depth-{}".format(d), c=colors[i], linestyle = "dotted")
            plt.plot(mean_eval/1, label="mean-eval-depth-{}".format(d), c=colors[i])
            
        plt.legend()
        plt.title(env_name)
        plt.xlabel("Viper Iteration")
        plt.grid()
        plt.savefig("plots/res_{}.pdf".format(env_name))
        plt.clf()