import os
import sys
# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import sumo_rl
from sumo_rl.models.util import *
from sumo_rl.agents.pg_multi_agent import PGMultiAgent
from sumo_rl.models.dcrnn_model import *
from sumo_rl.models.transformer_model import PolicyNetwork

import torch
import torch.optim as optim

# Set up the environment
NET_FILE = '../sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml'
ROUTE_FILE = '../sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml'
OUTPUT_CSV = '../results/my_result'
MODEL_DIR = '../models'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_env = sumo_rl.parallel_env(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    out_csv_name=OUTPUT_CSV,
    use_gui=False,
    num_seconds=3000,
    begin_time=100,
    fixed_ts=True,
    reward_fn="weighted_wait_queue"  # r_i,t = queue_i,t + gamma * wait_i,t
)

# Reset environment and get initial observations
# obs, _ = env.reset()

# transformer
# Build graph representation
traffic_signals = [ts for _, ts in train_env.aec_env.env.env.env.traffic_signals.items()]
max_lanes = max(len(ts.lanes) for ts in traffic_signals)  # max incoming lanes
max_green_phases = max(ts.num_green_phases for ts in traffic_signals)
ts_phases = [ts.num_green_phases for ts in traffic_signals]
feature_size = 2*max_lanes
hid_dim = 128
num_virtual_nodes = 2  # incoming/outgoing

ts_indx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = (
    construct_graph_representation(traffic_signals)) # adj_list: np.array [|E|, 2]
action_mask = create_action_mask(num_nodes, max_green_phases, ts_phases)


G = build_networkx_G(adj_list)  # creates an undirected graph
laplacian_matrix, eigenvals, eigenvecs = get_laplacian_eigenvecs(G)
edge_index = torch.tensor(adj_list.T, dtype=torch.long, device=DEVICE)  # [2, |E|]

k = 5
hops = 2

# transformer
model_args = {
    "laplacian_matrix": laplacian_matrix,
    "eigenvals": eigenvals,
    "eigenvecs": eigenvecs,
    "ts_indx": ts_indx,
    "in_features": feature_size,
    "output_features": max_green_phases,
    # Network hyperparameters
    "num_transformer_layers": 2,
    "num_proj_layers": 2,
    "hidden_features": hid_dim,
    "action_mask": action_mask
}


#PGMultiAgent = PGMultiAgent(ts_indx, edge_index, num_nodes, k, hops, model_args, DEVICE, model=None)
#PGMultiAgent.train(train_env, num_episodes=100, model_dir=MODEL_DIR)

#continue training block
pretrained_models = {}
for ts in traffic_signals:
    ts_id = ts.id
    model_path = f"../models/{ts_id}_model.pth"
    model = PolicyNetwork(model_args).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    pretrained_models[ts_id] = model
PGMultiAgent = PGMultiAgent(ts_indx, edge_index, num_nodes, k, hops, model_args, DEVICE, models=pretrained_models)
PGMultiAgent.train(train_env, num_episodes=100, model_dir=MODEL_DIR)

