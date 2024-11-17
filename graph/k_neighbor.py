import torch
import torch_geometric

def get_neighbors(ts_index, adj_list, k):
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

  return {ts_id: torch_geometric.utils.k_hop_subgraph(node_idx=node_index, 
                                                           num_hops=k, 
                                                           edge_index=adj_list, 
                                                           relabel_nodes=False) \
               for ts_id, node_index in ts_index.items()}

class LastKFeatures:
  '''
  Stores last k features for each TrafficSignal.
  '''

  def __init__(self, ts_id_list, feature_shape, k):
    '''
    Args:
      ts_id_list (list[str]): List of TrafficSignal ids.
      feature_shape (tuple(int)): shape of feature.
      k (int): k features to track
    '''
    self.k = k
    self.ts_features = {ts_id: [torch.zeros(feature_shape) for _ in range(k)] for ts_id in ts_id_list }

  def update(self, features):
    '''
    Update self.ts_features with most recent features.

    Args:
      features (dict{str: tensor}): dictionary mapping feature for each TrafficSignal id.
    '''

    for ts_id, feature in features.items():
      self.ts_features[ts_id].insert(0, feature)
      self.ts_features[ts_id].pop()

    

if __name__ == "__main__":
  ts_list = ['ts0', 'ts1', 'ts2']
  k = 2
  last_k = LastKFeatures(ts_list, (5,), k)
  print("STEP 1")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {'ts0': torch.ones(5), 'ts1': torch.ones(5), 'ts2': torch.ones(5)}
  last_k.update(features)
  print("STEP 2")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {'ts0': torch.full((5,), 2), 'ts1': torch.full((5,), 2), 'ts2': torch.full((5,), 2)}
  last_k.update(features)
  print("STEP 3")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {'ts0': torch.full((5,), 3), 'ts1': torch.full((5,), 3), 'ts2': torch.full((5,), 3)}
  last_k.update(features)
  print("STEP 4")
  print(f"ts_features: {last_k.ts_features}")
