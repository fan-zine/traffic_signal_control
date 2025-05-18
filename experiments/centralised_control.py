import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumo_rl.environment.env import SumoEnvironment
from sumo_rl.models.util import *
from sumo_rl.models.dcrnn_model import DCRNNEncoder, TLPhasePredictor, TSModel
from sumo_rl.agents.pg_single_agent import PGSingleAgent

# Constants
# 定义交通网络和车辆路径
NET_FILE = '../sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml'
ROUTE_FILE = '../sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml'
TRAIN_OUTPUT_CSV = '../results/train_result'
TEST_OUTPUT_CSV = '../results/test_result'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 交通信号灯数量
NUM_NODES = None
# TODO ？
NUM_VIRTUAL_NODES = 2  # incoming/outgoing nodes
# 最大车道数
MAX_LANES = None
# 最大绿灯相位数
# 相位数：交通信号灯在一个完整周期内可以切换的不同状态的数量，每个状态称为一个相位。比如一个交通信号灯的总周期为80s，绿灯45s，红灯30s，黄灯5s
# 相位1：南北方向绿灯，东西方向红灯；相位2：南北方向左转绿灯，东西方向红灯；相位3、4 东西方向同理
MAX_GREEN_PHASES = None
LEARNING_RATE = 0.001
THRESHOLD = 100
HID_DIM = 128

# Initialize environment
train_env = SumoEnvironment(
    net_file=NET_FILE,
    route_file=ROUTE_FILE,
    out_csv_name=TRAIN_OUTPUT_CSV,
    use_gui=False,
    num_seconds=3000,
    begin_time=100,
    fixed_ts=True,
    reward_fn="weighted_wait_queue"  # r_i,t = queue_i,t + gamma * wait_i,t
)

#observations = env.reset()  # [{ts_id: ts_observation_1}, ... ,{ts_id: ts_observation_k}]

# Build graph representation
# TODO 该看这了
traffic_signals = [ts for _, ts in train_env.traffic_signals.items()]
MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)  # max incoming lanes
MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
ts_phases = [ts.num_green_phases for ts in traffic_signals]

ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)

NUM_NODES = num_nodes

adj_mx = construct_binary_adjacency_matrix(traffic_signals, num_virtual_nodes=NUM_VIRTUAL_NODES, inlane_ts=incoming_lane_ts, outlane_ts=outgoing_lane_ts)

# Initialize models
encoder = DCRNNEncoder(
    input_dim=2 * MAX_LANES,  # Density and queue
    adj_mat=adj_mx,
    max_diffusion_step=5,
    hid_dim=HID_DIM,
    num_nodes=NUM_NODES,
    num_rnn_layers=1,
    filter_type="dual_random_walk"
).to(DEVICE)

action_mask = create_action_mask(NUM_NODES, MAX_GREEN_PHASES, ts_phases)
head = TLPhasePredictor(
    hid_dim=HID_DIM,
    input_dim=2 * MAX_LANES,
    num_nodes=NUM_NODES,
    num_virtual_nodes=NUM_VIRTUAL_NODES,
    max_green_phases=MAX_GREEN_PHASES,
    mask=action_mask
).to(DEVICE)

model = TSModel(encoder, head).to(DEVICE)

k = 5  # warmup-steps
# Initialize agent
agent = PGSingleAgent(
    actor=model,
    ts_idx=ts_idx,
    device=DEVICE,
    num_nodes=NUM_NODES,
    max_lanes=MAX_LANES,
    lr=LEARNING_RATE,
    k=k
)

# Training loop
print("Starting training...")
agent.train(train_env, num_episodes=50)

# Save the models
model_path = "../models/dcrnn.pth"
model_dir = os.path.dirname(model_path)
os.makedirs(model_dir, exist_ok=True)

agent.save_models(model_path)
print(f"Models saved at {model_path}.")

# Testing loop
#print("Starting testing...")
#agent.test(env, num_episodes=TEST_EPISODES)

