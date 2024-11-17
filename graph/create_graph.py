from .graph_converter import construct_graph_representation
from .node_features import batch_traffic_signal_feature
from ..sumo_rl.environment.traffic_signal import TrafficSignal

import torch_geometric

def construct_graph_and_features(ts_list, device):
  '''
  Returns:
    * node_features (Tensor): list of feature vectors for each node.
    * adj_list (Tensor): list of directed edges in graph.
    * ts_indx (dict[str: int]): mapping of TrafficSignal id to node index.
    * lane_indx (dict[str: int]): mapping of lane id to adj_list index.
    * num_nodes (int): Total number of nodes in graph.

  Args:
    * ts_list (list[TrafficSignal]): list of TrafficSignals.
    * device (str): Type of device for Tensor.
  '''
  ts_indx, num_nodes, lane_indx, adj_list = construct_graph_representation(ts_list, device)
  node_features = batch_traffic_signal_feature(ts_list, ts_indx, num_nodes, device)

  return node_features, adj_list, ts_indx, lane_indx, num_nodes


def get_neighbors(ts_indx, adj_list, k):
  '''
  Use k_hop_subgraph (https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.k_hop_subgraph).
  Returns dictionary with tuple representing subgraph:
    * subset (LongTensor): the nodes involved in the subgraph
    * edge_index (LongTensor): the filtered edge_index connectivity
    * mapping (LongTensor): the mapping from node indices in node_idx to their new location
    * edge_mask (BoolTensor): edge mask indicating which edges were preserved
  
  Args:
    * ts_indx (dict[str: int]): mapping of TrafficSignal id to node index.
    * adj_list (Tensor): list of directed edges in graph.
    * k (int): value of k, for k-hop neighbors.
  '''

  neighbors = {ts_id: torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
                                                           num_hops=k, 
                                                           edge_index=adj_list, 
                                                           relabel_nodes=False) \
               for ts_id, node_index in ts_indx.items()}

  return neighbors

