"""
Created on Thu Apr 13 16:26:44 2023

@author: ayhant
"""

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
    
    


# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=2000)
#         self.gamma = 0.95
#         self.epsilon = 1.0
#         self.epsilon_decay = 0.995
#         self.epsilon_min = 0.01
#         self.learning_rate = 0.001
#         self.model = self._build_model()
 
#     def _build_model(self):
#         model = Sequential() 
#         model.add(Dense(32, activation="relu",
#                         input_dim=self.state_size))
#         model.add(Dense(32, activation="relu"))
#         model.add(Dense(self.action_size, activation="linear"))
#         model.compile(loss="mse",
#                      optimizer=Adam(lr=self.learning_rate))
#         return model
 
#     def remember(self, state, action, reward, next_state, done): 
#         self.memory.append((state, action,
#                             reward, next_state, done))

#     def train(self, batch_size):
#          minibatch = random.sample(self.memory, batch_size)
#             for state, action, reward, next_state, done in minibatch:
#                 target = reward # if done 
#             if not done:
#                 target = (reward +
#                           self.gamma *
#                           np.amax(self.model.predict(next_state)[0]))
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0) 
#         if self.epsilon &amp;gt; self.epsilon_min:
#                 self.epsilon *= self.epsilon_decay

#     def act(self, state):
#         if np.random.rand() &amp;lt;= self.epsilon:
#                 return random.randrange(self.action_size) 
#         act_values = self.model.predict(state)
#             return np.argmax(act_values[0])

#     def save(self, name): 
#         self.model.save_weights(name)
        
        


#with cnn
class DDQN_Agent:
    #
    # Initializes attributes and constructs CNN model and target_model
    #
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        
        # Hyperparameters
        self.gamma = 0.99           # Discount rate
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.1      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.update_rate = 10000    # Number of steps until updating the target network
        
        # Construct DQN models
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.model.summary()

    #
    # Constructs CNN
    #
    def build_model(self):
        # model = Sequential()
        
        # # Conv Layers
        # model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        # model.add(Activation('relu'))
        
        # model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        # model.add(Activation('relu'))
        
        # model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        # model.add(Activation('relu'))
        # model.add(Flatten())

        # # FC Layers
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        
        # model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0))
        inputs = tf.keras.Input(shape=self.state_size)
        c1 = tf.keras.layers.Conv2D(32, (8, 8), strides=2, padding='same',)(inputs)
        c1a=tf.keras.layers.Activation(activations.relu)(c1)


        c2 = tf.keras.layers.Conv2D(64, (4, 4), strides=2, padding='same')(c1a)
        c2a=tf.keras.layers.Activation(activations.relu)(c2)


        c3 = tf.keras.layers.Conv2D(64, (3, 3), strides=1, padding='same')(c2a)
        c3a=tf.keras.layers.Activation(activations.relu)(c3)


        flatten_layer=tf.keras.layers.Flatten()(c3a)

        d1=tf.keras.layers.Dense(128)(flatten_layer)
        d1a=tf.keras.layers.Activation(activations.relu)(d1)


        d3=tf.keras.layers.Dense(64)(d1a)
        d3a=tf.keras.layers.Activation(activations.relu)(d3)

        d4=tf.keras.layers.Dense(self.action_size)(d3a)
        final_activation=tf.keras.layers.Activation(activations.linear)(d4)


        model = tf.keras.Model(inputs=inputs, outputs=final_activation)
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=None, decay=0.0))

        return model

    #
    # Stores experience in replay memory
    #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])  # Returns action using policy

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                max_action = np.argmax(self.model.predict(next_state)[0])
                target = (reward + self.gamma * self.target_model.predict(next_state)[0][max_action])
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)
        



def process_frame(frame):

    mspacman_color = np.array([210, 164, 74]).mean()
    img = frame[1:176:2, ::2]    # Crop and downsize
    img = img.mean(axis=2)       # Convert to greyscale
    img[img==mspacman_color] = 0 # Improve contrast by making pacman white
    img = (img - 128) / 128 - 1  # Normalize from -1 to 1.
    
    return np.expand_dims(img.reshape(88, 80, 1), axis=0)




def  blend_images (images, blend):
    avg_image = np.expand_dims(np.zeros((88, 80, 1), np.float64), axis=0)

    for image in images:
        avg_image += image
        
    if len(images) < blend:
        return avg_image / len(images)
    else:
        return avg_image / blend
    
    
    



# env = gym.make('MsPacman-v0')
# state_size = (88, 80, 1)
# action_size = env.action_space.n
# agent = DDQN_Agent(state_size, action_size)
# #agent.load('models/')

# episodes = 5000
# batch_size = 32
# skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
# total_time = 0   # Counter for total number of steps taken
# all_rewards = 0  # Used to compute avg reward over time
# blend = 4        # Number of images to blend
# done = False

# #main logic
# for e in range(episodes):
#     total_reward = 0
#     game_score = 0
#     state = process_frame(env.reset())
#     images = deque(maxlen=blend)  # Array of images to be blended
#     images.append(state)
    
#     for skip in range(skip_start): # skip the start of each game
#         env.step(0)
    
#     for time in range(20000):
#         env.render()
#         total_time += 1
        
#         # Every update_rate timesteps we update the target network parameters
#         if total_time % agent.update_rate == 0:
#             agent.update_target_model()
        
#         # Return the avg of the last 4 frames
#         state = blend_images(images, blend)
        
#         # Transition Dynamics
#         action = agent.act(state)
#         next_state, reward, done, _ = env.step(action)
        
#         game_score += reward
#         total_reward += reward
        
#         # Return the avg of the last 4 frames
#         next_state = process_frame(next_state)
#         images.append(next_state)
#         next_state = blend_images(images, blend)
        
#         # Store sequence in replay memory
#         agent.remember(state, action, reward, next_state, done)
        
#         state = next_state
        
#         if done:
#             all_rewards += game_score
            
#             print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
#                   .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
            
#             break
            
#         if len(agent.memory) > batch_size:
#             agent.replay(batch_size)