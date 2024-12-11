import torch
import numpy as np
from torch.nn import Linear, LeakyReLU, Sequential, ModuleList, LogSoftmax, BatchNorm1d
from torch_geometric.nn.conv import TransformerConv

class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.transformer = TransformerConv(
            in_channels=in_channels, out_channels=out_channels, heads=heads, dropout=dropout)
        self.norm = BatchNorm1d(num_features=out_channels)
        self.relu = LeakyReLU()
        self.proj = Linear(in_features=heads*out_channels,
                           out_features=out_channels)

    def forward(self, x, edge_index):
        x = self.transformer(x, edge_index)
        x = self.proj(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class PolicyNetwork(torch.nn.Module):
    def __init__(self, args):
        '''
        Args:
        * args (dict): model arguments
        '''
        super(PolicyNetwork, self).__init__()
        self.laplacian = args["laplacian_matrix"]
        self.eigenvalues = args["eigenvals"]
        self.eigenvecs = args["eigenvecs"]
        self.eigenvec_len = self.eigenvecs.shape[0]
        self.num_layers = args.get("num_transformer_layers", 2)
        self.num_proj_layers = args.get("num_proj_layers", 2)

        assert self.num_layers > 1
        assert self.num_proj_layers > 1

        self.input_features = args["in_features"]  # feature_size
        self.hidden_features = args["hidden_features"]
        self.output_features = args["output_features"]
        self.dropout = args.get("dropout", 0.0)
        self.mask = args["action_mask"]

        # Add LSTM for processing temporal features
        self.lstm = torch.nn.LSTM(input_size=self.input_features,
                                  hidden_size=self.hidden_features,
                                  num_layers=1,
                                  batch_first=True)

        # Add Transformer layers
        self.layers = torch.nn.ModuleList()
        self.layers.append(TransformerBlock(in_channels=self.hidden_features + self.eigenvec_len,
                                            out_channels=self.hidden_features, heads=4, dropout=self.dropout))
        prev = self.hidden_features
        for _ in range(self.num_layers - 2):
            self.layers.append(TransformerBlock(in_channels=self.hidden_features + prev +
                                                self.eigenvec_len, out_channels=self.hidden_features,
                                                heads=4, dropout=self.dropout))
            prev = self.hidden_features
        self.layers.append(TransformerConv(in_channels=self.hidden_features + prev +
                                           self.eigenvec_len, out_channels=self.hidden_features, heads=1,
                                           dropout=self.dropout))

        # Add projection head for classification
        proj_head = []
        for _ in range(self.num_proj_layers - 1):
            proj_head.append(
                Linear(in_features=self.hidden_features, out_features=self.hidden_features))
            proj_head.append(LeakyReLU())
        proj_head.append(Linear(in_features=self.hidden_features,
                                out_features=self.output_features))

        self.proj_head = Sequential(*proj_head)
        self.softmax = LogSoftmax()

    def forward(self, node_features, edge_index, agent_index, subgraph_indices):
        '''
        Args:
        * node_features: shape [num_timesteps, num_nodes, feature_size]
        * edge_index: edge index describing graph [2, |E|]
        * agent_index: agent to generate action for
        * subgraph_indices: list of node indices used in subgraph to add positional encoding
        '''
        num_timesteps, num_nodes, feature_size = node_features.shape

        # Reshape for LSTM: [num_nodes, num_timesteps, feature_size]
        x = node_features.permute(1, 0, 2)  # [num_nodes, num_timesteps, feature_size]

        # Pass through LSTM: [num_nodes, num_timesteps, feature_size] -> [num_nodes, hidden_features]
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take the last hidden state: [num_nodes, hidden_features]

        # Positional encoding for subgraph
        pos = np.take(self.eigenvecs, subgraph_indices.cpu().numpy(), axis=0)
        pos = torch.tensor(pos, dtype=torch.float32, device=x.device)

        #if pos.shape[0] != x.shape[0]:
        #    pos = torch.nn.functional.pad(
        #        pos, (0, 0, 0, x.shape[0] - pos.shape[0]))

        prev = None
        for layer in self.layers:
            temp = x
            if prev is not None:
                x = torch.cat([x, prev], dim=1)
            prev = temp
            x = torch.cat([x, pos], dim=1)
            x = layer(x, edge_index)

        local_agent_idx = (subgraph_indices == agent_index).nonzero(as_tuple=True)[0].item()

        logits = self.proj_head(x[local_agent_idx])
        logits = logits + (1 - self.mask[agent_index]) * -1e9

        return logits
