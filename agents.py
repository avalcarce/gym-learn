import numpy as np


class ExperienceReplayAgent:
    """
    All this agent does is manage a Experience Replay Memory.
    """
    def __init__(self, per_proportional_prioritization=False,
                 per_apply_importance_sampling=False, per_alpha=0.2, per_beta0=0.4):
        # Experience Replay parameters. See https://arxiv.org/pdf/1511.05952.pdf
        self.memory = None  # Experience replay memory
        self.per_proportional_prioritization = per_proportional_prioritization
        self.per_apply_importance_sampling = per_apply_importance_sampling
        self.per_alpha = per_alpha
        self.per_beta0 = per_beta0
        self.per_beta = per_beta0
        self.per_epsilon = 1E-6
        self.prio_max = 0

    def anneal_per_importance_sampling(self, step, max_step):
        if self.per_proportional_prioritization and self.per_apply_importance_sampling:
            self.per_beta = self.per_beta0 + step*(1-self.per_beta0)/max_step

    def error2priority(self, errors):
        return np.power(np.abs(errors) + self.per_epsilon, self.per_alpha)

    def save_experience(self, state, action, reward, state_next, done):
        if self.memory is not None:
            experience = (state, action, reward, state_next, done)
            if self.per_proportional_prioritization:
                self.memory.add(max(self.prio_max, self.per_epsilon), experience)
            else:
                self.memory.add(experience)
        else:
            raise ValueError("The Experience Replay memory is not initialized.")

    def retrieve_experience(self, batch_size):
        idx = None
        priorities = None
        w = None

        # Extract a batch of random transitions from the replay memory
        if self.per_proportional_prioritization:
            idx, priorities, experience = self.memory.sample(batch_size)
            if self.per_apply_importance_sampling:
                sampling_probabilities = priorities / self.memory.total()
                w = np.power(self.memory.n_entries * sampling_probabilities, -self.per_beta)
                w = w / w.max()
        else:
            experience = self.memory.sample(batch_size)

        return idx, priorities, w, experience


class AgentEpsGreedy(ExperienceReplayAgent):
    def __init__(self, n_actions, value_function_model, eps=1., per_proportional_prioritization=False,
                 per_apply_importance_sampling=False, per_alpha=0.2, per_beta0=0.4):
        ExperienceReplayAgent.__init__(self, per_proportional_prioritization=per_proportional_prioritization,
                                       per_apply_importance_sampling=per_apply_importance_sampling, per_alpha=per_alpha,
                                       per_beta0=per_beta0)

        # RL parameters
        self.n_actions = n_actions              # Number of actions the agent can do
        self.value_func = value_function_model  # Value Function object
        self.eps = eps                          # Probability of choosing a random action. Epsilon-Greedy policy
        self.current_value = None  # Current value of the value function (i.e. expected discounted return)
        self.explore = True                     # Whether to explore or not
        self.state = None                       # The current state the agent is on
        self.loss_v = 0                         # Loss value of the last training epoch

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

    @staticmethod
    def __format_experience(experience):
        states_b, actions_b, rewards_b, states_n_b, done_b = zip(*experience)
        states_b = np.array(states_b)
        actions_b = np.array(actions_b)
        rewards_b = np.array(rewards_b)
        states_n_b = np.array(states_n_b)
        done_b = np.array(done_b).astype(int)

        return states_b, actions_b, rewards_b, states_n_b, done_b

    def train_on_experience(self, batch_size, discount, double_dqn=False):
        loss_v = 0
        if self.memory is None:
            raise NotImplementedError("Please provide an Experience Replay memory.")

        if self.memory.n_entries >= batch_size:
            idx, priorities, w, experience = self.retrieve_experience(batch_size)
            states_b, actions_b, rewards_b, states_n_b, done_b = self.__format_experience(experience)

            if double_dqn:
                q_n_b = self.predict_q_values(states_n_b)  # Action values on the arriving state
                best_a = np.argmax(q_n_b, axis=1)
                q_n_target_b = self.predict_q_values(states_n_b, use_old_params=True)
                targets_b = rewards_b + (1. - done_b) * discount * q_n_target_b[np.arange(batch_size), best_a]
            else:
                q_n_b = self.predict_q_values(states_n_b, use_old_params=True)  # Action values on the next state
                targets_b = rewards_b + (1. - done_b) * discount * np.amax(q_n_b, axis=1)

            targets = self.predict_q_values(states_b)
            for j, action in enumerate(actions_b):
                targets[j, action] = targets_b[j]

            if self.per_apply_importance_sampling:
                loss_v, errors = self.train(states_b, targets, w=w)
            else:
                loss_v, errors = self.train(states_b, targets)
            errors = errors[np.arange(len(errors)), actions_b]

            if self.per_proportional_prioritization:  # Update transition priorities
                priorities = self.error2priority(errors)
                for i in range(batch_size):
                    self.memory.update(idx[i], priorities[i])
                self.prio_max = max(priorities.max(), self.prio_max)

        return loss_v
