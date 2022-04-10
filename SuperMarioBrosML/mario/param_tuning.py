#use this prior to running the smb.py file to create optomized hypertuning files for the learning
# Importing the optimzation frame - HPO
import optuna
# PPO algo for RL
from stable_baselines3 import PPO
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy
# Import the sb3 monitor for logging 
from stable_baselines3.common.monitor import Monitor
# Import the vec wrappers to vectorize and frame stack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# Import os to deal with filepaths
import os
# Import sys
import sys

#Import the game, joypad, and simplified controls
import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



LOG_DIR = './optimization/logs/'
OPT_DIR = './optimization/opt/'
SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(1))
SAVE_PATH_BEST_PARAMS = os.path.join(LOG_DIR, 'best_params.txt')
SAVE_PATH_BEST_TRIAL = os.path.join(LOG_DIR, 'best_trial.txt')
SAVE_PATH_ALL_TRIALS = os.path.join(LOG_DIR, 'all_trials.txt')

# For different ALGOs these parameters NEED to be changed
# Function to return test hyperparameters - define the object function
def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 64, 2048),
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-6, 1e-4),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }

# Run a training loop and return mean reward 
def optimize_agent(trial):
    try:
        #use this to avoid truncated mini-batches
        model_params = optimize_ppo(trial) 
        #remove truncated runs
        model_params['n_steps'] = model_params['n_steps']//64*64
        # Re-create live environment 
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        # Create algo 
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, device='cuda', **model_params)
        #should use a min of 100k
        #model.learn(total_timesteps=30000)
        model.learn(total_timesteps=100000)

        # Evaluate model 
        # n_eval_episodes is number of games to evaluate in. Increase this for better results
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)
        env.close()

        #save models
        SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
        model.save(SAVE_PATH)

        return mean_reward

    except Exception as e:
        return -1000

# Creating the experiment 
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=20, n_jobs=1)
#study.optimize(optimize_agent, n_trials=100, n_jobs=1)

best_param_results = study.best_params
best_trial_results = study.best_trial
all_trial_results = study.get_trials(deepcopy=True)

#redirect sysout to write to text files
sys.stdout = open(SAVE_PATH_BEST_PARAMS, "w")
print(best_param_results)
sys.stdout = open(SAVE_PATH_BEST_TRIAL, "w")
print(best_trial_results)
sys.stdout = open(SAVE_PATH_ALL_TRIALS, "w")
print(all_trial_results)