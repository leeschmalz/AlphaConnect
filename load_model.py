import gym
from stable_baselines3 import PPO
from Env import Env
import time
data_path = "BTC_ETH_LTC_Jan12000_May82022.csv"
symbols = ["BTC","ETH","LTC"]


models_dir = "models\PPO_1652059017"

env = Env(data_path,symbols)
env.reset()

model_path = f"{models_dir}/latest_opponent.zip"
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(action)
        print(rewards)