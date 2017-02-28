import numpy as np


class AgentEpsGreedy:
    def __init__(self, n_actions, value_function_model, eps=1., summaries_path_current=None):
        self.n_actions = n_actions
        self.value_func = value_function_model
        self.eps = eps
        self.summaries_path_current = summaries_path_current
        self.current_value = None  # Current value of the value function (i.e. expected discounted return)

    def act(self, state):
        action_values = self.value_func.predict([state])[0]

        policy = np.ones(self.n_actions) * self.eps / self.n_actions
        a_max = np.argmax(action_values)
        policy[a_max] += 1. - self.eps

        a = np.random.choice(self.n_actions, p=policy)
        self.current_value = action_values[a]
        return a

    def train(self, states, targets, w=None):
        loss, errors = self.value_func.train(states, targets, w=w)
        return loss, errors

    def predict_q_values(self, states, use_old_params=False):
        return self.value_func.predict(states, use_old_params)
