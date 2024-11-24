import torch
from torch.nn import Linear, LeakyReLU, Sequential, ModuleList, LogSoftmax
from torch_geometric.nn.conv import TransformerConv

class PolicyNetwork(torch.nn.Module):
  def __init__(self, ts_map, traffic_system, args, aggr="mean"):
        '''
        Args:
        * ts_map (Dict{str: int}): mapping of traffic signal id to node index
        * traffic_system (Tensor[2,|E|]): adjacency list of entire traffic system
        * args (dict): arguments relating to traffic system. e.i: eigenvectors.
        * aggr (str): type of aggregation to use.
        '''
        super(PolicyNetwork, self).__init__()
        self.traffic_system = traffic_system
        self.ts_map = ts_map
        self.laplacian = args["laplacian_matrix"]
        self.eigenvecs = args["eigenvecs"]
        self.num_layers = args.get("num_transformer_layers", 2)
        self.num_proj_layers = args.get("num_proj_layers", 2)

        assert self.num_layers > 1
        assert self.num_proj_layers > 1


        self.input_features = args["in_features"]
        self.hidden_features = args["hidden_features"]
        self.output_features = args["output_features"]
        # Add Transformer layers
        self.layers = ModuleList()
        self.layers.append(TransformerConv(in_channels=self.input_features, out_channels=self.hidden_features, heads=4))
        self.layers.append(LeakyReLU())
        for _ in range(self.num_layers-2):
          self.layers.append(TransformerConv(in_channels=self.input_features, out_channels=self.hidden_features, heads=4))
          self.layers.append(LeakyReLU())
        self.layers.append(TransformerConv(in_channels=self.hidden_features, out_channels=self.hidden_features, heads=4))

        # Add MLP for classification
        proj_head = [Linear(in_features=2*self.hidden_features, out_features=self.hidden_features), LeakyReLU()]
        for _ in range(self.num_proj_layers-2):
          proj_head.append(Linear(in_features=self.hidden_features, out_features=self.hidden_features))
          proj_head.append(LeakyReLU())
        proj_head.append(Linear(in_features=self.hidden_features, out_features=self.output_features))

        self.proj_head = Sequential(*proj_head)
        self.softmax = LogSoftmax()

  def forward(self, node_features, edge_index, num_green_phases):
    # Add work
    pass
  
