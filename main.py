import gymnasium as gym
import sumo_rl
import os
from custom_reward import custom_reward

NET_FILE = './sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml'
ROUTE_FILE = './sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml'
OUTPUT_CSV = './output_csv/output.csv'
env = sumo_rl.parallel_env(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                out_csv_name=OUTPUT_CSV,
                use_gui=True,
                num_seconds=3000,
                reward_fn=custom_reward
                )
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents} 
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)
    print(observations)