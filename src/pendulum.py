import tensorflow as tf
import numpy as np
import gym

from actors import QActor
from tf2_model import load_model, save_model
from session import session

model_file = 'models/pendulum'


def make_nn(state_space, action_space):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=4000, activation="relu", input_shape=(state_space,)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=1000, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(units=2000, activation="relu"))
    model.add(tf.keras.layers.Dense(units=action_space, activation="linear"))
    return model


def pendulum_end(env, next_state, time_step):
    return time_step >= 800


def pendulum_reward_manip(reward, next_state, time_step):
    return reward


def simple_saver(actor, episode, replay_buffer):
    save_model(actor.get_model(), model_file)


def random_pd_trainer(replay_buffer, batch_size=20000):
    ids = np.random.randint(replay_buffer.size,
                            size=min(replay_buffer.size, batch_size))
    return replay_buffer.get_numpy_array()[ids]


def pendulum_solved(time_step, replay_buffer):
    return False


def pendulum_actor_action(action):
    if action == 0:
        return [-2.0]
    else:
        return [2.0]


def pendulum_action_discretize(action):
    if action[0] >= 0.0:
        return 1
    else:
        return 0


def pendulum(replay=True, load_net=True):
    randomness = 0.5
    env = gym.make('Pendulum-v0')
    action_space_size = 2
    state_space_size = env.observation_space.shape[0]

    model = load_model(model_file) if load_net else make_nn(state_space_size,
                                                          action_space_size)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0008), loss="mean_squared_error")
    actor = QActor(q_learning_model=model,
                   state_space_size=state_space_size,
                   action_space_size=action_space_size,
                   gamma=0.95)
    if replay:
        randomness = 0.0

    session(env=env, actor=actor, episodes=13,
            stop_f=pendulum_end,
            reward_manip_f=pendulum_reward_manip,
            save_f=simple_saver,
            trainer=random_pd_trainer,
            solved_f=pendulum_solved,
            train=(not replay),
            rand_decay=0.99,
            randomness=randomness,
            alter_actor_action=pendulum_actor_action,
            action_prepare_to_save=pendulum_action_discretize)


pendulum(True)
