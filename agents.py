import numpy as np


class AgentEpsGreedy:
    def __init__(self, n_actions, value_function_model, eps=1., per_proportional_prioritization=False,
                 per_apply_importance_sampling=False, per_alpha=0.2, per_beta0=0.4, max_step=1E6):
        # RL parameters
        self.n_actions = n_actions              # Number of actions the agent can do
        self.value_func = value_function_model  # Value Function object
        self.eps = eps                          # Probability of choosing a random action. Epsilon-Greedy policy
        self.current_value = None  # Current value of the value function (i.e. expected discounted return)
        self.explore = True                     # Whether to explore or not
        self.max_step = max_step                # How many steps the agent will execute
        self.state = None                       # The current state the agent is on
        self.loss_v = 0                         # Loss value of the last training epoch
        self.total_reward = 0                   # Variable to accumulate the rewards collected so far.

        # Experience Replay parameters
        self.memory = None  # Experience replay memory
        self.per_proportional_prioritization = per_proportional_prioritization
        self.per_apply_importance_sampling = per_apply_importance_sampling
        self.per_alpha = per_alpha
        self.per_beta0 = per_beta0
        self.per_beta = per_beta0

    def act(self, state=None):
        """
        Choose an action.
        :param state:   ndarray describing the state the action will be chosen on. If not provided, the agent will act
                        from the current state.
        :return: An integer denoting the chosen action.
        """

        # Input check
        if state is None:
            if self.state is not None:
                state = self.state
            else:
                raise TypeError("Missing 1 required positional argument when the agent's state is unknown: 'state'")

        # Evaluate actions
        action_values = self.value_func.predict([state])[0]
        a_max = np.argmax(action_values)

        if self.explore:
            policy = np.ones(self.n_actions) * self.eps / self.n_actions
            policy[a_max] += 1. - self.eps
            a = np.random.choice(self.n_actions, p=policy)
        else:
            a = a_max

        self.current_value = action_values[a]
        return a

    def train(self, states, targets, w=None):
        loss, errors = self.value_func.train(states, targets, w=w)
        self.loss_v = loss
        return loss, errors

    def predict_q_values(self, states, use_old_params=False):
        return self.value_func.predict(states, use_old_params)

    def anneal_per_importance_sampling(self, step):
        if self.per_proportional_prioritization and self.per_apply_importance_sampling:
            self.per_beta = self.per_beta0 + step*(1-self.per_beta0)/self.max_step
