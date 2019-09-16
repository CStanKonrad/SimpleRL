from collections import deque
import numpy as np


class ReplayBuffer:
    def default_adjust(buf):
        while buf.size > buf.max_size:
            buf.forget()

    def __init__(self, max_size, adjust_f=default_adjust):
        self.storage = deque()
        self.max_size = max_size
        self.size = 0
        self.adjust_f = adjust_f

    def remember(self, x):
        self.storage.append(x)
        self.size += 1
        self.adjust_f(self)

    def forget(self):
        self.storage.pop_left()
        self.size -= 1

    def get_numpy_array(self):
        return np.array(self.storage)


def random_trainer(replay_buffer, batch_size=10000):
    ids = np.random.randint(replay_buffer.size,
                            size=min(replay_buffer.size, batch_size))
    return replay_buffer.get_numpy_array()[ids]


def empty_saver(actor, episode, replay_buffer):
    return None


def identity(x):
    return x


def session(env, actor, episodes, solved_f,
            stop_f, reward_manip_f, save_f=empty_saver,
            replay_buffer_size=50000, trainer=random_trainer,
            rand_decay=1.0, randomness=0.0,
            train=True,
            alter_actor_action=identity,
            action_prepare_to_save=identity):
    replay_buffer = ReplayBuffer(replay_buffer_size)

    for episode in range(episodes):
        state = env.reset()
        end = False

        time_step = 0
        random_behaviour = np.random.uniform() < randomness
        randomness *= rand_decay
        while not end:
            mode = 0 if random_behaviour else 1

            action = alter_actor_action(actor.decide(np.array([state]), mode=mode)[0])

            next_state, reward, _, _ = env.step(action)

            reward = reward_manip_f(reward, next_state, time_step)

            if time_step % 20 == 0:
                print('########')
                if random_behaviour:
                    print('random exploration:')
                print('episode:', episode, ' time_step:', time_step,
                      '\n\tstate:', state, ' action:', action,
                      '\n\treward:', reward, ' next_state:', next_state)
                print('\n\tq(state):', actor.predict_q(np.array([state])))

            action = action_prepare_to_save(action)

            sars = np.append(np.append(state, action), np.append(np.array([reward]), next_state))
            replay_buffer.remember(sars)

            time_step += 1

            if solved_f(time_step, replay_buffer):
                print('solved')
                return

            state = next_state
            end = stop_f(env, next_state, time_step)
            env.render()

        if train:
            sars = trainer(replay_buffer)
            # print(sars)
            actor.train_model(sars)
            save_f(actor, episode, replay_buffer)
