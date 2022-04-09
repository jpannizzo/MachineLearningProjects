'''
TOTAL TIMESTEPS RUN SO FAR: 5,600,000
TOTAL TIME RUN: ~19 hrs
'''

#Import the game, joypad, and simplified controls
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

#Import GreayScaling Wrapper
from gym.wrappers import GrayScaleObservation
#Import Vectorization Wrappers and Frame Stacker Wrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of framestacking
from matplotlib import pyplot as plt

#Import RL dependencies
#file management
import os
# import PPO algo
from stable_baselines3 import PPO
# import base callbacks for saving models
from stable_baselines3.common.callbacks import BaseCallback
# import for improved saving model code
from TrainAndLoggingCallback import TrainAndLoggingCallback

#location of saved models
CHECKPOINT_DIR = './train/'
#tensorflow log files. can see progress with tensor board
LOG_DIR = './logs/'

#setup model saving callback
#saves every XX,000 steps
callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)

#Setup Environment
#Create base env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#simplify controls of env
"""
SIMPLE_MOVEMENT reduces action commands to the following
[
    ['NOOP'], #No Action
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left']
]
"""
env = JoypadSpace(env, SIMPLE_MOVEMENT)
#grayscale. need keep_dim=True for frame stacking
env = GrayScaleObservation(env, keep_dim=True)
#wrap in dummy env
env = DummyVecEnv([lambda: env])
#stack the frames
env = VecFrameStack(env, 4, channels_order='last')

#create PPO model
#CnnPolicy is very good at processing images
#   tensorboard_log keeps the logs for use in tensorflow
#   learning_rate is very important. longer time is more stable where shorter time may create an unreliable AI. need to look into more
#   n_steps = frames to wait per game before we update neural network. need to look into more and tinker with
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

#MlpPolicy is very good for tabular data XLS, CSV, JSON data. It could still be used as the neural network for the game.
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512)

#update learning rate. typically between 0.1 and 0.000001. 
#custom_objects = { 'learning_rate': 0.00001}

d = os.path.dirname(os.getcwd())

#comment out CnnPolicy line above and use following for loading already saved data
model = PPO.load(d+"\\completed\\completed_PPO_6_cont_RuntimeSteps_4250000")
model.set_env(env)
#train the AI model
model.learn(total_timesteps=5000000, callback=callback)
model.save('/train/latestmodel')

#run the game to show the latest model
#need to break/stop to end the process
state = env.reset()
while True:

    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()