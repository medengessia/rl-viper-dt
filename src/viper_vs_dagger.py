import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
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
        model = algo_rl.load(path_to_expert)
    else:
        model.learn(n_iter) # Training the policy

    return model


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
            s, r, term, trunc, _ = env.step(action)
            done = term or trunc
            tot += r
        avg += tot
    return avg / n_iter


if __name__ == "__main__":

    algo_rl = DQN
    nb_data_from_nn_policy = 10_000

    for regul_type in ["max_depth", "max_leaf_nodes"]:

        for k, env_name in enumerate(["Acrobot-v1", "CartPole-v1", "MountainCar-v0"]):

            env = gym.make(env_name)
            path_to_expert = '/content/drive/MyDrive/Colab Notebooks/Research Project/policies/' + env_name + '.zip'
            model = get_policy_nn(env, algo_rl, nb_data_from_nn_policy, path_to_expert)
            perf_expert = get_perf_expert(env, model)
            print(perf_expert)

            colors = ["black", "red", "blue", "green"]

            for i, d in enumerate([2,3,4,5]):
                mean_eval_dag = 0
                mean_eval_vip = 0             

                if regul_type == "max_leaf_nodes":
                    l_nodes = 2**d - 1

                    for seed in range(5):
                        list_eval_dag = np.load("/content/drive/MyDrive/Colab Notebooks/Research Project/experiments_dagger/{}/{}/{}/seed{}/list_eval.npy".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed))[:-2]
                        list_eval_vip = np.load("/content/drive/MyDrive/Colab Notebooks/Research Project/experiments_viper/{}/{}/{}/seed{}/list_eval.npy".format(regul_type, env_name, regul_type + str(l_nodes).rjust(3, '0'), seed))[:-2]
                        
                      
                        if env_name == "Acrobot-v1":
                            list_eval_dag_norm = (list_eval_dag+500)/(perf_expert+500)
                            list_eval_vip_norm = (list_eval_vip+500)/(perf_expert+500)

                        elif env_name == "MountainCar-v0":
                            list_eval_dag_norm = (list_eval_dag+200)/(perf_expert+200)
                            list_eval_vip_norm = (list_eval_vip+200)/(perf_expert+200)

                        else:
                            list_eval_dag_norm = list_eval_dag/perf_expert
                            list_eval_vip_norm = list_eval_vip/perf_expert

                        mean_eval_dag = mean_eval_dag + list_eval_dag_norm
                        mean_eval_vip = mean_eval_vip + list_eval_vip_norm

                    plt.plot(mean_eval_dag/5, label="dagger-eval-leaf_nodes-{}".format(l_nodes), c=colors[i], linestyle = "dotted")
                    plt.plot(mean_eval_vip/5, label="viper-eval-leaf_nodes-{}".format(l_nodes), c=colors[i])

                else:

                    for seed in range(5):
                        list_eval_dag = np.load("/content/drive/MyDrive/Colab Notebooks/Research Project/experiments_dagger/{}/{}/{}/seed{}/list_eval.npy".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed))[:-2]
                        list_eval_vip = np.load("/content/drive/MyDrive/Colab Notebooks/Research Project/experiments_viper/{}/{}/{}/seed{}/list_eval.npy".format(regul_type, env_name, regul_type + str(d).rjust(2, '0'), seed))[:-2]
                        
                      
                        if env_name == "Acrobot-v1":
                            list_eval_dag_norm = (list_eval_dag+500)/(perf_expert+500)
                            list_eval_vip_norm = (list_eval_vip+500)/(perf_expert+500)

                        elif env_name == "MountainCar-v0":
                            list_eval_dag_norm = (list_eval_dag+200)/(perf_expert+200)
                            list_eval_vip_norm = (list_eval_vip+200)/(perf_expert+200)

                        else:
                            list_eval_dag_norm = list_eval_dag/perf_expert
                            list_eval_vip_norm = list_eval_vip/perf_expert

                        mean_eval_dag = mean_eval_dag + list_eval_dag_norm
                        mean_eval_vip = mean_eval_vip + list_eval_vip_norm

                    plt.plot(mean_eval_dag/5, label="dagger-eval-depth-{}".format(d), c=colors[i], linestyle = "dotted")
                    plt.plot(mean_eval_vip/5, label="viper-eval-depth-{}".format(d), c=colors[i])

            file_path = os.path.join("/content/drive/MyDrive/Colab Notebooks/Research Project/viper_vs_dagger/{}/{}".format(regul_type, env_name))
            os.makedirs(file_path, exist_ok=True)  

            plt.legend()
            plt.title(env_name)
            plt.xlabel("Iteration")
            plt.grid()
            plt.savefig("/content/drive/MyDrive/Colab Notebooks/Research Project/viper_vs_dagger/{}/{}/vip_vs_dag_{}_{}.pdf".format(regul_type, env_name, env_name, regul_type)) # 1 ENV + 1 regul type
            plt.clf()