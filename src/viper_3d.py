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


def get_data_from_datasets(datasets, nb_data_from_nnpolicy):
    """Extracts a dataset for training a decision tree from a list of datasets based on their reward probabilities.

    Args:
        datasets (Ndarray): An array of datasets.
        nb_data_from_nnpolicy (int): Number of iterations for data generating functions.

    Returns:
        Ndarray: The chosen dataset.
    """
    indices = datasets.shape[0]

    probabilities = datasets[:,-1]/datasets[:,-1].sum()
    
    probabilities = np.where(probabilities == 0, 1e-10, probabilities)
    probabilities = probabilities/probabilities.sum()

    dataset_dt_indices = np.random.choice(np.arange(indices), nb_data_from_nnpolicy, replace=False, p=probabilities)

    dataset_dt = datasets[dataset_dt_indices]

    return dataset_dt


def fit_dtrees(data, algo_dt, depth=10):
    """Fits a decision tree to the policy-generated dataset and returns it with its score.

    Args:
        data (Ndarray): The dataset to fit.
        algo_dt : A decision tree algorithm.
        depth (int, optional): The desired maximal depth. Defaults to 10.

    Returns:
        list: A list of trained decicision trees.
    """
    X, y, rewards = data[:,:-2], data[:,-2], data[:,-1] # Extracting the states as input features and the actions as labels

    clf = algo_dt(random_state=0)
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas

    d_trees = []
    for ccp_alpha in ccp_alphas:
        d_tree = algo_dt(max_depth=depth, random_state=0, ccp_alpha=ccp_alpha)
        d_tree.fit(X, y) # Training each decision tree on the RL dataset
        d_trees.append(d_tree)

    #print("Score of fitted DT is {}".format(d_tree.score(X,y)))

    return d_trees


def eval(dt, mdp, n_iter=10):
    """Evaluates a decision tree by giving its mean cumulative reward.

    Args:
        dt (tree): The decision tree to evaluate.
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem).
        n_iter (int, optional): Number of iterations. Defaults to 10.

    Returns:
        float: The mean cumulative reward of a tree over the iterations.
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
    list_list_scores = []
    list_number_of_nodes = []
    list_dtrees = []
    policy = get_policy_nn(mdp, algo_rl, nb_data_from_nnpolicy, path_to_expert)

    for i in range(iter_viper):
        #print('iteration {}'.format(i))

        data = generate_data(mdp, policy, nb_data_from_nnpolicy, reward_mode)

        if i > 0:
            datasets = np.vstack((datasets, data))
        else:
            datasets = data
        
        dataset_dt = get_data_from_datasets(datasets, nb_data_from_nnpolicy)

        d_trees = fit_dtrees(dataset_dt, algo_dt)
        list_dtrees.append(d_trees)

    for list_dtree in list_dtrees:

        list_scores = []
        number_of_nodes = []
        
        for tree in list_dtree:
            score, nb_nodes = eval(tree, mdp), tree.tree_.node_count
            list_scores.append(score)
            number_of_nodes.append(nb_nodes)

        list_list_scores.append(list_scores)
        list_number_of_nodes.append(number_of_nodes)

    return list_number_of_nodes, list_list_scores