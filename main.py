from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM, Flatten
from keras.layers import Embedding
from keras.engine import Input
from keras.optimizers import RMSprop, SGD
from keras.models import Model
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l1l2, activity_l1l2
import keras.models

from utils import *
from datetime import datetime

import ple
import ple.games.flappybird
import os
import pygame.display

import io
import sys
import time

if sys.version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle


const_is_playback = False
const_is_debug = False

const_episode_epoch_frames = 100000
const_frames_back = 30
epsilon_explore = 0.2
const_epsilon_explore_decay = 2e-7
const_frames_max = 10000
const_lr = 1e-6

const_rl_gamma = 0.99

const_l1_regularization = 0.001

if const_is_debug:
    const_episode_epoch_frames = 10000


version = 'v4'
init_log('./logs/rl-{0}-{1}.log'.format(version, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
file_csv_loss = open('./results/loss-{}.csv'.format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")), 'a')

os.environ['SDL_AUDIODRIVER'] = "waveout"

if not const_is_playback:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.display.init()
    screen = pygame.display.set_mode((1,1))

def process_state(state):
    return np.array( list(state.values()) ).astype(np.float32)

def get_action(q_values):
    global actions_available, dimensions_actions

    if np.random.random() < epsilon_explore:
        if const_is_playback:
            logging.info('random action')

        rand_index = np.random.randint(0, dimensions_actions)
        q_values = np.zeros((dimensions_actions,)).tolist()
        q_values[rand_index] = 1.0
        return actions_available[rand_index], q_values

    if const_is_playback:
        logging.info('{} {}'.format(q_values[0], q_values[1]))

    return actions_available[np.argmax(q_values)], q_values


game = ple.games.flappybird.FlappyBird()
env = ple.PLE(game, display_screen=const_is_playback, state_preprocessor=process_state)
env.init()

dimensions_state = env.getGameStateDims()[0] + 1 #prev action
actions_available = env.getActionSet()
dimensions_actions = len(actions_available)

logging.info('model build started')

model = Sequential()

if(os.path.exists('model-{}.h5'.format(version))):
    model = keras.models.load_model('model-{}.h5'.format(version))
    logging.info('model loaded')
else:
    #model.add(LSTM(output_dim=100, input_dim=dimensions_state, input_length=const_frames_back, return_sequences=True,
    #               W_regularizer=l1(const_l1_regularization), U_regularizer=l1(const_l1_regularization)))
    #model.add(LSTM(output_dim=100, W_regularizer=l1(const_l1_regularization),
    #               U_regularizer=l1(const_l1_regularization)))
    #model.add(Flatten())

    model.add(TimeDistributed(Dense(500, activation='relu', W_regularizer=l1l2(const_l1_regularization), b_regularizer=l1l2(const_l1_regularization)), input_shape=(const_frames_back, dimensions_state)))

    model.add(Flatten())

    model.add(Dense(500, activation='relu', W_regularizer=l1l2(const_l1_regularization), b_regularizer=l1l2(const_l1_regularization)))

    model.add(Dense(output_dim=dimensions_actions, W_regularizer=l1l2(const_l1_regularization),
                    activity_regularizer=activity_l1l2(const_l1_regularization)))

    optimizer = SGD(lr=const_lr)
    model.compile(loss='mse', optimizer=optimizer)

    logging.info('model build finished')

x = np.zeros((const_frames_back, dimensions_state)).tolist()
prev_action = 0

x_buffer = []
q_buffer = []
r_buffer = []

epoch = 0
episode = 0
total_epoch = 0
frames_total = 0

while True:
    episode += 1

    if frames_total > const_episode_epoch_frames and len(x_buffer) > 1:
        epoch += 1
        frames_total= 0

        y_buffer = []

        for i in range(len(x_buffer)):
            y_index = np.argmax(q_buffer[i])
            y_val = q_buffer[i]

            y_val[y_index] = r_buffer[i]

            if i < len(x_buffer) - 1:
                x_next = x_buffer[i+1]
                x_input = np.reshape(np.array(x_next), (1, const_frames_back, dimensions_state))
                q_values = model.predict(x_input, batch_size=1)
                action, q_values = get_action(q_values)

                y_val[y_index] = r_buffer[i] + const_rl_gamma * np.max(q_values)

            y_buffer.append(y_val)

        history = model.fit(np.array(x_buffer), np.array(y_buffer), batch_size=32, nb_epoch=1)

        for loss in history.history['loss']:
            total_epoch += 1
            file_csv_loss.write('{};{};{};{};{}\n'.format(total_epoch, episode, loss, np.max(r_buffer), np.average(r_buffer)))
            logging.info('epoch: {} episode: {} loss: {} max: {} avg: {}'.format(epoch, episode, loss, np.max(r_buffer), np.average(r_buffer)))

        model.save('model-{}.h5'.format(version))

        file_csv_loss.flush()

        x_buffer = []
        r_buffer = []
        q_buffer = []
        prev_action = 0
        x = np.zeros((const_frames_back, dimensions_state)).tolist()


    for frame in range(const_frames_max):
        frames_total += 1
        if env.game_over():
            break

        state = env.getGameState().tolist()
        state.append(prev_action + 1) #zero index = NAN
        x.append(state)
        x = x[-const_frames_back:]

        x_input = np.reshape(np.array(x), (1, const_frames_back, dimensions_state))
        q_values = model.predict(x_input, batch_size=1)
        q_values = np.reshape(q_values, (dimensions_actions,))

        action, q_values = get_action(q_values)

        reward = env.act(action) + 0.002

        q_buffer.append(q_values)
        x_buffer.append(x[:])
        r_buffer.append(reward)

        if const_is_playback:
            time.sleep(0.03)

    if const_is_playback:
        logging.info('game over')

    env.reset_game()

    epsilon_explore -= const_epsilon_explore_decay
    epsilon_explore = max(epsilon_explore, 0.01)