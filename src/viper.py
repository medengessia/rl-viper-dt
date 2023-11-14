import numpy as np

def get_policy_nn(mdp, algo_rl, n_iter, path_to_expert):
    """Generates a RL policy by training a RL Neural Network and returns the policy.

    Args:
        mdp (gym.Env): a Markov Decision Process (Reinforcement Learning Problem).
        algo_rl : A RL Algorithm.
        n_iter (int): Number of iterations.
        path_to_expert (str): a path to an existing policy.

    Returns:
        The obtained policy.
    """
    policy = algo_rl('MlpPolicy', mdp, verbose=1)

    if path_to_expert is not None:
        policy = algo_rl.load(path_to_expert)
    else:
        policy.learn(n_iter) # Training of the policy

    return policy


def generate_data(mdp, policy, n_iter, reward_mode):
    """Generates data, with actions weighted by their reward.

    Args:
        mdp (gym.Env): a Markov Decision Process (Reinforcement Learning Problem).
        policy : A RL policy.
        n_iter (int): Number of iterations.
        reward_mode (str): The way of associating a reward to a (S,a) couple between 'cumulative', 'instant' and 'uniform'.

    Returns:
        Ndarray: The obtained dataset.
    """
    s, _ = mdp.reset()
    X, y, rewards = np.zeros((n_iter, mdp.observation_space.shape[0])), np.zeros((n_iter,1)), np.zeros((n_iter,1)) # Initialization of the dataset
    r_T = 0    # The computed reward for one trajectory

    start = 0  # Initialization of the first index with a particular reward
    for i in range(n_iter):
        X[i] = s   # Current state

        action, _ = policy.predict(s, deterministic=True) # Entering a state

        new_s, reward, terminated, truncated, infos = mdp.step(action) # Taking action
        s = new_s # Getting a new state

        y[i] = action   # The chosen action

        if reward_mode == 'instant':
            rewards[i] = reward

        if reward_mode == 'uniform':
                rewards[i] = np.random.uniform(0, 1)

        r_T += reward

        if terminated or truncated:  # At this point, r_T is the cumulated reward along the episode
            s, _ = mdp.reset()

            if reward_mode == 'cumulative':
                rewards[start:i+1] = r_T
                start = i + 1   # The variable start takes the value of the next index
            
            r_T = 0

    dataset = np.hstack((X,y))
    dataset = np.hstack((dataset,rewards))

    return dataset


def get_data_from_datasets(datasets):
    """Extracts a dataset for training a decision tree from a list of datasets based on their reward probabilities.

    Args:
        datasets (Ndarray): An array of datasets.

    Returns:
        Ndarray: The chosen dataset.
    """
    indices = datasets.shape[0]

    probabilities = datasets[:,-1]/datasets[:,-1].sum()
    
    if any(x == 0 for x in probabilities):
        probabilities = np.where(probabilities == 0, 1e-10, probabilities)
        probabilities = probabilities/probabilities.sum()

    dataset_dt_indices = np.random.choice(np.arange(indices), 1000, replace=False, p=probabilities)

    dataset_dt = datasets[dataset_dt_indices]

    return dataset_dt


def fit_dt(data, algo_dt, depth=5):
    """Fits a decision tree to the policy-generated dataset and returns it with its score.

    Args:
        data (Ndarray): The dataset to fit.
        algo_dt : A decision tree algorithm.
        depth (int, optional): The desired maximal depth. Defaults to 5.

    Returns:
        tree: The trained decicision tree.
    """
    X, y, rewards = data[:,:-2], data[:,-2], data[:,-1] # Extracting the states as input features and the actions as labels
    d_tree = algo_dt(max_depth=depth)
    d_tree.fit(X,y) # Training the decision tree on the RL dataset 
    #print("Score of fitted DT is {}".format(d_tree.score(X,y)))

    return d_tree


def eval(dt, mdp, n_iter=10):
    """Evaluates a decision tree by giving its mean cumulative reward.

    Args:
        dt (tree): The decision tree to evaluate.
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem).
        n_iter (int, optional): Number of iterations. Defaults to 10.

    Returns:
        float: The mean cumulative reward over the iterations.
    """
    score_mean = 0

    for i in range(n_iter):
        r_traj = 0
        done = False
        s, _ = mdp.reset()

        while not done:
            action = dt.predict([s]) # Entering a state
            new_s, reward, terminated, truncated, infos = mdp.step(int(action[0])) # Seeing how the decision tree imitates a RL agent

            s = new_s # Getting a new state
            r_traj += reward
            done = terminated or truncated
        score_mean += r_traj
    
    return score_mean/n_iter


def Viper(mdp, algo_dt, algo_rl, iter_viper, nb_data_from_nnpolicy, reward_mode='cumulative', path_to_expert=None):
    """Proposes an implementation of Viper algorithm.

    Args:
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem).
        algo_dt : A decision tree algorithm.
        algo_rl : A RL Algorithm.
        iter_viper (int): Number of iterations for VIPER.
        nb_data_from_nnpolicy (int): Number of iterations for data generating functions.
        reward_mode (str, optional): The way of associating a reward to a (S,a) couple between 'cumulative', 'instant' and 'uniform'. Defaults to 'cumulative'.
        path_to_expert (str, optional): A path to an existing policy. Defaults to None.

    Returns:
        Ndarray: The mean cumulative rewards of the trees.
    """
    scores = np.zeros(iter_viper)
    policy = get_policy_nn(mdp, algo_rl, nb_data_from_nnpolicy, path_to_expert)

    for i in range(iter_viper):
        #print('iteration {}'.format(i))

        data = generate_data(mdp, policy, nb_data_from_nnpolicy, reward_mode)

        if i > 0:
            datasets = np.vstack((datasets, data))
        else:
            datasets = data
        
        dataset_dt = get_data_from_datasets(datasets)

        dt = fit_dt(dataset_dt, algo_dt)
        scores[i] = eval(dt, mdp)

    return scores





#### Just in case

# def choose_best_dt(dt_list, mdp, n_iter=5_000):
#     """Returns the top-scored decision tree and its rewards.

#     Args:
#         dt_list (list): A list of decision trees coupled with their scores.
#         mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem).
#         n_iter (int): Number of exploration iterations. Defaults to 5_000.

#     Returns:
#         tuple: The best decision tree and its rewards.
#     """
#     scores = [dt_list[i][1] for i in range(len(dt_list))]
#     index = scores.index(max(scores))

#     best_tree = dt_list[index][0]  # Taking the best trained tree
#     rewards = np.zeros((n_iter,1))  # Preparing the vessel of its rewards

#     s, _ = mdp.reset()
#     r_T = 0    # The computed reward for one trajectory

#     start = 0  # Initialization of the first index with a particular reward
#     for i in range(n_iter):
        
#         action = best_tree.predict([s]) # Entering a state
#         new_s, reward, terminated, truncated, infos = mdp.step(int(action[0])) # Seeing how a decision tree imitates a RL agent
#         s = new_s # Getting a new state
#         r_T += reward

#         if terminated or truncated: # At this point, r_T is the cumulated reward along the episode
#             s, _ = mdp.reset()
#             rewards[start:i+1] = r_T
#             start = i + 1   # The variable start takes the value of the next index
#             r_T = 0
        
#     return best_tree, rewards




# best_tree = dt_list[0]
#     best_reward = 0

#     for dt in dt_list: # Computing the total reward for each decision tree

#         s, _ = mdp.reset()
#         sum_rewards = 0

#         for i in range(n_iter):
            
#             action = dt.predict([s]) # Entering a state
#             new_s, reward, terminated, truncated, infos = mdp.step(int(action[0])) # Seeing how a decision tree imitates a RL agent
#             s = new_s # Getting a new state
#             sum_rewards += reward

#             if terminated or truncated:
#                 s, _ = mdp.reset()
        
#         if sum_rewards > best_reward: # Updating the best reward and the best tree
#             best_reward = sum_rewards
#             #print("New best reward is {}".format(best_reward))
#             best_tree = dt