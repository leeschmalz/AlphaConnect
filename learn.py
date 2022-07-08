import gym
from torch import flatten
from stable_baselines3 import PPO,A2C,DQN
from sb3_contrib import RecurrentPPO
import os
from Env import Env
import time
import random

policy = 'MlpLstmPolicy'
n_layers_range = [2,3,4,5,10]
layer_size_range = [32,64,128,256,512]
batch_size_range = [64,128,256,512]
gae_lambda_range = [0.9,0.95,0.99]
learning_rate_range = [0.003,0.001,0.0001,0.00001]
n_steps_range = [128,256,512,1024]

TIMESTEPS = 1000000
logdir = "logs"
while True:
    # choose random parameters
    n_layers = random.choice(n_layers_range)
    layer_size = random.choice(layer_size_range)
    batch_size = random.choice(batch_size_range)
    gae_lambda = random.choice(gae_lambda_range)
    learning_rate = random.choice(learning_rate_range)
    n_steps = random.choice(n_steps_range)

    print('n_layers',n_layers)
    print('layer_size',layer_size)
    print('batch_size',batch_size)
    print('gae_lambda',gae_lambda)
    print('learning_rate',learning_rate)
    print('n_steps',n_steps)
    print('\n')
    
    name = f"HYPERSEARCH_RecurrentPPO_nl{n_layers}_ls{layer_size}_bs{batch_size}_gl{gae_lambda}_lr{learning_rate}_{int(time.time())}"
    models_dir = f"models/{name}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = Env(verbose=2,
            opponent='greedy',
            veritical_win_reward=20,
            exploration_reward=0,
            block_vertical_win_reward=0,
            flatten=True)

    env.reset()
    model = RecurrentPPO(
            "MlpLstmPolicy",
            env,
            n_steps=n_steps,
            learning_rate=learning_rate,
            tensorboard_log=logdir,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
            policy_kwargs=dict(
                net_arch=[dict(vf=[layer_size]*n_layers)],
                lstm_hidden_size=layer_size
                )
            )

    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=True, tb_log_name=name)
    
    #model.save(f"{models_dir}/{TIMESTEPS*i}")
    model.save(f"{models_dir}/final_model.zip")
    model.save(f"./models/latest_model.zip")
    