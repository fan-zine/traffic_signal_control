import torch
from .traffic_signal import TrafficSignal

# create graph representation, with TS nodes, and lanes as nodes
# return: ts_nodes, lanes, adj_list
# ts: list of TrafficSignal 
def construct_graph_representation(ts_list: list[TrafficSignal]):
    # collect traffic signal ids
    sort_ts_func = lambda ts: ts.ts_id
    ts_idx = {ts.ts_id: i for i, ts in enumerate(sorted(ts_list, key=sort_ts_func))}

    # collect all lane ids
    lanes = [ts.lanes + ts.out_lanes for ts in ts_list]
    lanes = [lane for ln in lanes for lane in ln]
    lanes_index = {ln_id: i for i,ln_id in enumerate(sorted(set(lanes)))}

    # calculate all ts_start, ts_end for all lanes
    adj_list = [[-1 for _ in range(2)] for _ in range(len(lanes_index))]


    # fill with additional dummy nodes
    for ts in ts_list:
        ts_id = ts_idx[ts.ts_id]
        for in_edge in ts.lanes:
            in_edge_idx = lanes_index[in_edge]
            adj_list[in_edge_idx][1] = ts_id
    
        for out_edge in ts.out_lanes:
            out_edge_idx = lanes_index[out_edge]
            adj_list[out_edge_idx][0] = ts_id

        print(ts.out_lanes)

    next_indx = len(ts_idx)
    # for unassigned positions, add dummy nodes
    for lane in adj_list:
        if lane[0] == -1 or lane[1] == -1:
            pos = 0
            if lane[1] == -1:
                pos = 1
            lane[pos] = next_indx
            next_indx += 1

    return ts_idx, lanes_index, torch.Tensor(adj_list)
    
if __name__ == "__main__":
    a = TrafficSignal("A0", ["A0_0", "A0_1"], ["A1_1", "A1_0", "A1_2"])
    a2 = TrafficSignal("A1", ["A1_0", "A1_1"], ["A0_0", "A2_0", "A0_1"])

    b,c,d = construct_graph_representation([a,a2])

    print(f"ts_idx: {b}")
    print(f"lanes_index: {c}")
    print(f"adj_list: {d}")
