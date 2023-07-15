import os
from environment import CustomEnv
import time
from stable_baselines3 import DQN
from stable_baselines3 import PPO
models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = CustomEnv()
env.reset()

model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
for i in range(1,30):
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")