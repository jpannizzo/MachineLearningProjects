This is the second agent trained using the full version of super mario brothers v0.
It ran 8,650,000 timesteps which took ~43 hours.
Parameters were found using HPO using the optuna library. 
Folders contain each sessions Tensorboard results. PPO_2 is the initial run and PPO_3 is the final run.
load the Zip of the Agent in order to view the results.

Some Notes:
Agent_2 performed better at earlier timesteps than Agent_1. However, training regressed and mario struggled to clear the 2nd pipe for long periods of time.
In hindsight, I believe taking the peak training data and re-running HPO once an agent regresses for 500k-1million consistent timesteps would increase performance.
This could be automated and dynamic parameters could be calculated and loaded prior to learning sessions.
Even without automation, manually updating the parameters using an HPO should create a better learning environment. 
The tradeoff being the amount of time needed to run hyper parameterized testing every time mean rewards decline for extended periods of time.
Ultimately, Agent_1 performed better over the same period of time with a significantly lower training rate and n_step.
