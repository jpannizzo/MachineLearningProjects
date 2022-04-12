# import for improved callback
from TrainAndLoggingCallback import TrainAndLoggingCallback
#import PPO training
from stable_baselines3 import PPO
#import our doom env
from doom import *
#import environment checker to check that env is valid
from stable_baselines3.common import env_checker

CHECKPOINT_DIR = './train/train_basic'
LOG_DIR = './logs/log_basic'

callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

#create non-rendered env
env = VizDoomGym()
#environment checker. used to verify that the custom environment is working properly
#env_checker.check_env(env)

#PPO model
#ToDo: update learning rate and n_steps with HPO.
#   n_steps = batch size for the model. If XXXX is the number of n_steps then XXXX observations, actions, log probabilities, and values will be stored in the buffer for 1 iteration.
#learning rate and n_steps is arbitrary in this example
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=256)

# Evaluate mean reward for 10 games. adds mean_reward and mean_length to tensorboard
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

#train the AI model & save last model
model.learn(total_timesteps=100000, callback=callback)
model.save('./train/latestmodel')