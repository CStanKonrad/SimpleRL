import tensorflow as tf
import numpy as np
import gym

from actors import QActor
from tf2_model import load_model, save_model
from session import session


model_file = 'models/cartpole'

def make_nn(state_space, action_space):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=24, activation="relu", input_shape=(state_space,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=16, activation="relu"))
    model.add(tf.keras.layers.Dense(units=24, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(units=action_space, activation="linear"))
    return model


def cart_pole_end(env, next_state, time_step):
    return abs(next_state[0]) >= 5.0 \
        or time_step >= 1000 or abs(next_state[2]) > 2.0


def cart_pole_reward_manip(reward, next_state, time_step):
    return reward - abs(next_state[0])/10.0 - abs(next_state[2])/3.0


def simple_saver(actor, episode, replay_buffer):
    save_model(actor.get_model(), model_file)


def random_cp_trainer(replay_buffer, batch_size=1000):
    ids = np.random.randint(replay_buffer.size,
                            size=min(replay_buffer.size, batch_size))
    return replay_buffer.get_numpy_array()[ids]


def cartpole_solved(time_step, replay_buffer):
    return time_step >= 500


def cartpole_actor_action_alter(action):
    if np.random.uniform() <= 0.1:
        action = np.random.randint(2)
    return action


def cart_pole(replay=True):
    randomness = 0.5
    env = gym.make('CartPole-v1')
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]

    model = load_model(model_file) if replay else make_nn(state_space_size,
                                                          action_space_size)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mean_squared_error')
    actor = QActor(q_learning_model=model,
                   state_space_size=state_space_size,
                   action_space_size=action_space_size,
                   gamma=0.9999)

    if replay:
        randomness = 0.0

    session(env=env, actor=actor, episodes=2000,
            stop_f=cart_pole_end,
            reward_manip_f=cart_pole_reward_manip,
            save_f=simple_saver,
            trainer=random_cp_trainer,
            solved_f=cartpole_solved,
            train=(not replay),
            rand_decay=0.992,
            randomness=randomness,
            alter_actor_action=cartpole_actor_action_alter)

cart_pole(True) #125 episodes
