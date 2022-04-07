#Import the game, joypad, and simplified controls
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

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
model_num = input("Model Number: ")
#load model
model = PPO.load('./train/best_model_'+model_num)

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

state = env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()