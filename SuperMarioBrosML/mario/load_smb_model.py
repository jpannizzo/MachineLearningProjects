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
from gym.core import RewardWrapper
# import PPO algo
from stable_baselines3 import PPO
# import base callbacks for saving models
from stable_baselines3.common.callbacks import BaseCallback
# import for improved saving model code
from TrainAndLoggingCallback import TrainAndLoggingCallback

#input model number at X's
#model_num = input("Model Number: ")
#load model
model = PPO.load('./train/best_model_50000')

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

state = env.reset()
while True:

    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()