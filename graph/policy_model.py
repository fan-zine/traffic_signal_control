import torch

class PolicyNetwork(torch.nn.Module):
  def __init__(self, ts_map, traffic_system, args, aggr="mean"):
        super(PolicyNetwork, self).__init__()
        self.traffic_system = traffic_system
        self.ts_map = ts_map
        self.laplacian = args["laplacian_matrix"]
        self.eigenvecs = args["eigenvecs"]

        # Add Transformer layers

        # Add MLP for classification

  def forward(self, node_features, edge_index, num_green_phases):
    # Add work
    pass
  
