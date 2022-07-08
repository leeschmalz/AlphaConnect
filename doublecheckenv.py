from Env import Env
import time

env = Env()
episodes = 1

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        time.sleep(1)
        random_action = env.action_space.sample()
        print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        print(env.board)
        print('reward',reward)
        print('done',done)
        print('info',info)
        print('\n')