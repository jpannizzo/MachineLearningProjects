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

d = os.path.dirname(os.getcwd())

#load model
#Change path to specific file
#model = PPO.load(d+'\\milestones\\Agent_1\\Agent_1')
model = PPO.load('./train/best_model_4900000')

#Setup Environment
#Need to duplicate the training env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()
while True:
 
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()