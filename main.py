import gymnasium as gym
import sumo_rl
import os
from custom_reward import custom_reward
# Add imports here

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


node_features, adj_list, ts_indx, lane_indx, num_nodes = construct_graph_and_features(env.ts_list, "cuda")
k = 2
last_k_features = LastKFeatures([i for i in range(num_nodes)], node_features[0].shape, k)
last_k_features.update({ node_index: node_features[node_index] for _, node_index in ts_indx.items()})

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents} # model(last_k_features, adj_list, env.action_space(agent))
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)
    print(observations)
    node_features = batch_traffic_signal_feature(env.ts_list, ts_indx, num_nodes, "cuda")
    last_k_features.update({ node_index: node_features[node_index] for _, node_index in ts_indx.items()})
