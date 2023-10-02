import numpy as np


def get_policy_nn(mdp, algo_rl, n_iter):
    """Generates a RL policy by training a RL Neural Network and returns the policy.

    Args:
        mdp (gym.Env): a Markov Decision Process (Reinforcement Learning Problem)
        algo_rl (abc.ABCMeta): A RL Algorithm
        n_iter (int): Number of exploration iterations

    Returns:
        stable_baselines3.dqn.dqn.DQN: The obtained policy
    """
    policy = algo_rl('MlpPolicy', mdp, verbose=1)
    policy.learn(n_iter) # Training of the policy

    return policy


def generate_data(mdp, policy, n_iter):
    """Generates data, with actions weighted by their reward.

    Args:
        policy (stable_baselines3.dqn.dqn.DQN): A policy
        n_iter (int): Number of exploration iterations

    Returns:
        tuple: The obtained dataset
    """
    s, _ = mdp.reset()
    X, y = np.zeros((n_iter, mdp.observation_space.shape[0])), np.zeros(n_iter) # Initialization of the dataset

    for i in range(n_iter):
        X[i] = s   # Current state
        action, _ = policy.predict(s, deterministic=True)
        new_s, reward, terminated, truncated, infos = mdp.step(action) # Moving to a new state
        s = new_s
        y[i] = action*reward   # The chosen action, weighted by its reward

        if terminated or truncated:
            s, _ = mdp.reset()

    return X, y


def fit_dt(data, algo_dt, depth=5):
    """Fits a decision tree to the policy-generated dataset.

    Args:
        data (tuple): The dataset to fit
        algo_dt (abc.ABCMeta): A decision tree algorithm
        depth (int): The desired maximal depth. Defaults to 5.

    Returns:
        sklearn.tree._classes.DecisionTreeClassifier: The trained classifier
    """
    X, y = data[0], data[1] # Extracting the states as input features and the actions as labels
    d_tree = algo_dt(max_depth=depth)
    d_tree.fit(X,y) # Training of the decision tree 

    return d_tree


def choose_best_dt(dt_list, mdp, n_iter=10_000):
    """Returns the decision tree with the best reward.

    Args:
        dt_list (list): A list of decision trees
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem)
        n_iter (int): Number of exploration iterations. Defaults to 10_000.

    Returns:
        sklearn.tree._classes.DecisionTreeClassifier: The best decision tree.
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
            best_tree = dt
        
    return best_tree


def Viper(mdp, algo_dt, algo_rl, n_iter_exploit, n_iter_explore):
    """Proposes an implementation of Viper algorithm.

    Args:
        mdp (gym.Env): A Markov Decision Process (Reinforcement Learning Problem)
        algo_dt (abc.ABCMeta): A decision tree algorithm
        algo_rl (abc.ABCMeta): A RL Algorithm
        n_iter_exploit (int): Number of exploitation iterations
        n_iter_explore (int): Number of exploration iterations

    Returns:
        sklearn.tree._classes.DecisionTreeClassifier: The best decision tree to evaluate a policy.
    """

    policy = get_policy_nn(algo_rl, mdp, n_iter_explore)
    dt_list = []

    for i in range(n_iter_exploit):

        data = generate_data(policy, mdp, n_iter_explore)
        dt = fit_dt(data, algo_dt)
        dt_list.append(dt)

    return choose_best_dt(dt_list, mdp)