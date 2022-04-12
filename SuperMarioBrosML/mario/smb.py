'''
TOTAL TIMESTEPS RUN SO FAR: 5000000
TOTAL TIME RUN: 32 hrs
'''
#for string to dict
import ast

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
#Display for monitoring
from stable_baselines3.common.monitor import Monitor
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy

#Import RL dependencies
#file management
import os
# import PPO algo
from stable_baselines3 import PPO
# import base callbacks for saving models
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
# import for improved saving model code
from TrainAndLoggingCallback import TrainAndLoggingCallback

#location of saved models
CHECKPOINT_DIR = './train/'
#tensorflow log files. can see progress with tensor board
LOG_DIR = './logs/'
#optimization dir
OPT_DIR = './optimization/opt/'

#Setup Environment
#Create base env
env = gym_super_mario_bros.make('SuperMarioBros-v0')
#for monitoring log dir
env = Monitor(env, LOG_DIR)
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

#setup model saving callback
#saves every XX,000 steps. send env for eval callback
callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIR)
#setup eval callback to evaluate model as it runs
#When using HPO like optuna the model loaded in will have episode length mean and episode reward mean from the study that occured in param_tuning.py 
#Those kick in at ~15k timesteps so this eval callback might not be necessary OR might be better to make them less frequent and increase the number of eval episodes
#eval_callback = EvalCallback(env, log_path=LOG_DIR, eval_freq=100000, deterministic=True, render=False, n_eval_episodes=10)
#callback list
#callback_list = CallbackList([callback, eval_callback])

d = os.path.dirname(os.getcwd())

#load optimization study
#param_file = d+"\\mario\\optimization\\logs\\best_params.txt"
#with open(param_file, 'r') as f:
#    string_params = f.read()
#convert from string to dict
#model_params = ast.literal_eval(string_params)
#remove truncated runs
#model_params['n_steps'] = model_params['n_steps']//64*64

#create PPO model
'''
CnnPolicy is very good at processing images
   tensorboard_log keeps the logs for use in tensorflow
   learning_rate is very important. longer time is more stable where shorter time may create an unreliable AI. need to look into more
   n_steps = frames to wait per game before we update neural network. need to look into more and tinker with
   device = cuda -> forces gPU usage instead of CPU. Should work by default and remove this if using CPU 
'''
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512, device='cuda',)

#MlpPolicy is very good for tabular data XLS, CSV, JSON data. It could still be used as the neural network for the game.
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, n_steps=512, device='cuda',)

#update learning rate. typically between 0.1 and 0.000001. 
#custom_objects = { 'learning_rate': 0.00001}

#create initial model + model_params
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, device='cuda', **model_params)
#load opt model, replace this for continued training
model = PPO.load("C:\\repos\MachineLearningProjects\SuperMarioBrosML\mario\completed\Agent2\cont_PPO_2_TimestepsRun_4900000", env)

#add evalution so mean reward/mean len is tracked for non PPO models
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

#train the AI model
model.learn(total_timesteps=3650000, callback=callback)
model.save('./train/latestmodel')

#run the game to show the latest model
#need to break/stop to end the process
state = env.reset()
while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()