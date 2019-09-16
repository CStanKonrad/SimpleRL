import numpy as np
from tf2_model import model_fit


def expectation_max(predictions):
    return np.max(predictions, axis=1)


def expectation_soft_max(predictions):
    return np.log(np.sum(np.exp(predictions, axis=1)))


def decision_max(predictions):
    return np.argmax(predictions, axis=1)


# probability of taking action a is e^Q(s,a) / e^V(s)
def decision_exp(predictions):
    state_value = expectation_soft_max(predictions)
    predictions = np.exp(predictions)
    prob = np.exp(predictions) / np.exp(state_value).reshape(-1, 1)

    cumulative_distribution = np.cumsum(prob, axis=1)
    rand = np.random.uniform(0.0, 1.0, prob.shape[0]).reshape(-1, 1)

    # np.argmax chooses smallest id among ids of max values
    chosen_actions = np.argmax((cumulative_distribution - rand) > 0.0, axis=1)
    return chosen_actions


class QActor:
    def __init__(self, q_learning_model,
                 state_space_size, action_space_size,
                 gamma=0.99,
                 fit_q_model_f=model_fit,
                 expectation_f=expectation_max,
                 decision_f=decision_max):
        self.q_learning_model = q_learning_model

        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.gamma = gamma

        self.fit_q_model_f = fit_q_model_f
        self.expectation_f = expectation_f
        self.decision_f = decision_f

    def predict_q(self, states):
        return self.q_learning_model.predict(states)

    def expectation(self, states):
        predictions = self.predict_q(states)
        return self.expectation_f(predictions)

    def decide(self, states, mode=1):
        if mode == 0:
            return np.random.randint(self.action_space_size, size=states.shape[0])
        else:
            predictions = self.predict_q(states)
            return self.decision_f(predictions)

    def train_model(self, sars):
        states = sars[:, 0:self.state_space_size]
        actions = sars[:, self.state_space_size]
        rewards = sars[:, self.state_space_size + 1]
        next_states = sars[:, (self.state_space_size + 2):]

        expectation = self.expectation(next_states)

        update = rewards + self.gamma * expectation

        new = self.predict_q(states)
        # print(new)
        # print(update)
        # print(actions)

        new[np.arange(update.shape[0]), actions.astype(int)] = update

        # print(new)

        self.fit_q_model_f(self.q_learning_model, states, new)

    def get_model(self):
        return self.q_learning_model
