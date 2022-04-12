#import our doom env
from doom import *
# Import eval policy to test agent
from stable_baselines3.common.evaluation import evaluate_policy
#import PPO
from stable_baselines3 import PPO

#create rendered env
env = VizDoomGym(render=True)

#load model
model = PPO.load('./train/latestmodel')

# Evaluate mean reward for 10 games
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

episodes = 10
for episode in range(episodes):
    obs = env.reset
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))