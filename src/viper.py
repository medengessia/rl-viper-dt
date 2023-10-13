import numpy as np


def get_policy_nn(mdp, algo_rl, n_iter, path_to_expert=None):
    """Generates a RL policy by training a RL Neural Network and returns the policy.

    Args:
        mdp (gym.Env): a Markov Decision Process (Reinforcement Learning Problem).
        algo_rl : A RL Algorithm.
        n_iter (int): Number of iterations.
        path_to_expert (str): a path to an existing policy. Defaults to None.

    Returns:
        The obtained policy.
    """
    policy = algo_rl('MlpPolicy', mdp, verbose=1)

    if path_to_expert is not None:
        policy = algo_rl.load(path_to_expert)
    else:
        policy.learn(n_iter) # Training of the policy

    return policy


def generate_data(mdp, policy, n_iter):
    """Generates data, with actions weighted by their reward.

    Args:
        mdp (gym.Env): a Markov Decision Process (Reinforcement Learning Problem).
        policy : A RL policy.
        n_iter (int): Number of iterations.

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
        r_T += reward

        if terminated or truncated:  # At this point, r_T is the cumulated reward along the episode
            s, _ = mdp.reset()
            rewards[start:i+1] = r_T
            start = i + 1   # The variable start takes the value of the next index
            r_T = 0 

    dataset = np.hstack((X,y))
    dataset = np.hstack((dataset,rewards))

    return dataset


def get_data_from_datasets(datasets):
    """Extracts a dataset for training a decision tree from a list of datasets based on their biggest reward,
    with respect to a probability distribution.

    Args:
        datasets (list): A list of datasets.

    Returns:
        Ndarray: The chosen dataset.
    """
    datasets = np.array(datasets)
    indices = datasets.shape[0]

    sum_max = np.sum([np.amax(datasets[i][:,5]) for i in range(len(datasets))])
    distributions = [np.amax(datasets[i][:,5])/sum_max for i in range(len(datasets))]

    dataset_dt_indices = np.random.choice(indices, 1, p=distributions)
    dataset_dt = datasets[list(dataset_dt_indices)].squeeze()

    return dataset_dt


def fit_dt(data, algo_dt, depth=5):
    """Fits a decision tree to the policy-generated dataset.

    Args:
        data (Ndarray): The dataset to fit.
        algo_dt : A decision tree algorithm.
        depth (int): The desired maximal depth. Defaults to 5.

    Returns:
        The trained decicision tree.
    """
    X, y, rewards = data[:,:4], data[:,4], data[:,5] # Extracting the states as input features and the actions as labels
    d_tree = algo_dt(max_depth=depth)
    d_tree.fit(X,y) # Training of the decision tree 
    print("Score of fitted DT is {}".format(d_tree.score(X,y)))

    return d_tree


def choose_best_dt(dt_list, mdp, n_iter=5_000):
    """Returns the decision tree with the best reward.

    Args:
        dt_list (list): A list of decision trees.
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem).
        n_iter (int): Number of exploration iterations. Defaults to 5_000.

    Returns:
        The best decision tree.
    """
    best_tree = dt_list[0]
    best_reward = 0

    for dt in dt_list: # Computing the total reward for each decision tree

        s, _ = mdp.reset()
        sum_rewards = 0

        for i in range(n_iter):
            
            action = dt.predict([s]) # Entering a state
            new_s, reward, terminated, truncated, infos = mdp.step(int(action[0])) # Seeing how a decision tree imitates a RL agent
            s = new_s # Getting a new state
            sum_rewards += reward

            if terminated or truncated:
                s, _ = mdp.reset()
        
        if sum_rewards > best_reward: # Updating the best reward and the best tree
            best_reward = sum_rewards
            print("New best reward is {}".format(best_reward))
            best_tree = dt
        
    return best_tree


def Viper(mdp, algo_dt, algo_rl, iter_viper, nb_data_from_nnpolicy, path_to_expert=None):
    """Proposes an implementation of Viper algorithm.

    Args:
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem).
        algo_dt : A decision tree algorithm.
        algo_rl : A RL Algorithm.
        iter_viper (int): Number of iterations for VIPER.
        nb_data_from_nnpolicy (int): Number of iterations for data generating functions.
        path_to_expert (str): A path to an existing policy. Defaults to None.

    Returns:
        The best decision tree to evaluate a policy.
    """
    datasets = []
    policy = get_policy_nn(mdp, algo_rl, nb_data_from_nnpolicy, path_to_expert)
    dt_list = []

    for i in range(iter_viper):
        print('iteration {}'.format(i))

        data = generate_data(mdp, policy, nb_data_from_nnpolicy)

        datasets.append(data)
        dataset_dt = get_data_from_datasets(datasets)

        dt = fit_dt(dataset_dt, algo_dt)
        dt_list.append(dt)

    return choose_best_dt(dt_list, mdp)