import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
    reward_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint


class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h, e, graph, mode="residual", edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          graph: torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    if not sparse:
      batch_size, num_nodes, hidden_dim = h.shape
    else:
      batch_size = None
      num_nodes, hidden_dim = h.shape
    h_in = h
    e_in = e

    # Linear transformations for node update
    Uh = self.U(h)  # B x V x H

    if not sparse:
      Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H
    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    Ah = self.A(h)  # B x V x H, source
    Bh = self.B(h)  # B x V x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    if not sparse:
      e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
    else:
      e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H

    # Update node features
    h = Uh + self.aggregate(Vh, graph, gates, edge_index=edge_index, sparse=sparse)  # B x V x H

    # Normalize node features
    if not sparse:
      h = self.norm_h(
          h.view(batch_size * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, hidden_dim) if self.norm_h else h
    else:
      h = self.norm_h(h) if self.norm_h else h

    # Normalize edge features
    if not sparse:
      e = self.norm_e(
          e.view(batch_size * num_nodes * num_nodes, hidden_dim)
      ).view(batch_size, num_nodes, num_nodes, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h = F.relu(h)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h = h_in + h
      e = e_in + e

    return h, e

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    Vh = gates * Vh  # B x V x V x H

    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if not sparse:
      if (mode or self.aggregation) == "mean":
        return torch.sum(Vh, dim=2) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
      elif (mode or self.aggregation) == "max":
        return torch.max(Vh, dim=2)[0]
      else:
        return torch.sum(Vh, dim=2)
    else:
      sparseVh = SparseTensor(
          row=edge_index[0],
          col=edge_index[1],
          value=Vh,
          sparse_sizes=(graph.size(0), graph.size(1))
      )

      if (mode or self.aggregation) == "mean":
        return sparse_mean(sparseVh, dim=1)

      elif (mode or self.aggregation) == "max":
        return sparse_max(sparseVh, dim=1)

      else:
        return sparse_sum(sparseVh, dim=1)


class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    if x.shape[2] == 2:
      y_embed = x[:, :, 0]
      x_embed = x[:, :, 1]
      if self.normalize:
        # eps = 1e-6
        y_embed = y_embed * self.scale
        x_embed = x_embed * self.scale

      dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
      dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

      pos_x = x_embed[:, :, None] / dim_t
      pos_y = y_embed[:, :, None] / dim_t
      pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
      pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
      pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    
    if x.shape[2] == 3:
      x_embed = x[:, :, 0]
      y_embed = x[:, :, 1]
      z_embed = x[:, :, 2]
      if self.normalize:
        # eps = 1e-6
        x_embed = x_embed * self.scale
        y_embed = y_embed * self.scale

      num_pos_feats_cord = self.num_pos_feats//2
      num_pos_feats_label = self.num_pos_feats

      dim_t_cord = torch.arange(num_pos_feats_cord, dtype=torch.float32, device=x.device)
      dim_t_label = torch.arange(num_pos_feats_label, dtype=torch.float32, device=x.device)
      
      dim_t_cord = self.temperature ** (2.0 * (torch.div(dim_t_cord, 2, rounding_mode='trunc')) / num_pos_feats_cord)
      dim_t_label = self.temperature ** (2.0 * (torch.div(dim_t_label, 2, rounding_mode='trunc')) / num_pos_feats_label)

      pos_x = x_embed[:, :, None] / dim_t_cord
      pos_y = y_embed[:, :, None] / dim_t_cord
      pos_z = z_embed[:, :, None] / dim_t_label
      pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
      pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
      pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)
      pos = torch.cat((pos_x, pos_y, pos_z), dim=2).contiguous()
    
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    if x.dim() == 3:
      x_embed = x
      dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
      dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
      pos_x = x_embed[:, :, :, None] / dim_t
      pos = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    
    elif x.dim() == 4 and x.shape[3] == 2:
      x_embed = x[:, :, :, 0]
      y_embed = x[:, :, :, 1]
      
      dim_t = torch.arange(self.num_pos_feats//2, dtype=torch.float32, device=x.device)
      dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)
  
      pos_x = x_embed[:, :, :, None] / dim_t
      pos_y = y_embed[:, :, :, None] / dim_t
      pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
      pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
      pos = torch.cat((pos_x, pos_y), dim=3).contiguous()
    
    return pos


class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    if len(x.shape) == 1:
      x_embed = x
      dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
      dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

      pos_x = x_embed[:, None] / dim_t
      pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
      return pos_x
    
    elif len(x.shape) == 2:
      if x.shape[1] == 2:
        num_pos_feats = self.num_pos_feats//2
        x_embed = x[:, 0]
        y_embed = x[:, 1]

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / num_pos_feats)

        pos_x = x_embed[ :, None] / dim_t
        pos_y = y_embed[ :, None] / dim_t
        pos_x = torch.stack((pos_x[ :, 0::2].sin(), pos_x[ :, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[ :, 0::2].sin(), pos_y[ :, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_x, pos_y), dim=1).contiguous()
        return pos
      
      if x.shape[1] == 3:
        num_pos_feats_cord = self.num_pos_feats//4
        num_pos_feats_label = self.num_pos_feats//2
        
        x_embed = x[:, 0]
        y_embed = x[:, 1]
        z_embed = x[:, 2]

        dim_t_cord = torch.arange(num_pos_feats_cord, dtype=torch.float32, device=x.device)
        dim_t_label = torch.arange(num_pos_feats_label, dtype=torch.float32, device=x.device)
        dim_t_cord = self.temperature ** (2.0 * (torch.div(dim_t_cord, 2, rounding_mode='trunc')) / num_pos_feats_cord)
        dim_t_label = self.temperature ** (2.0 * (torch.div(dim_t_label, 2, rounding_mode='trunc')) / num_pos_feats_label)

        pos_x = x_embed[ :, None] / dim_t_cord
        pos_y = y_embed[ :, None] / dim_t_cord
        pos_z = z_embed[ :, None] / dim_t_label
        
        pos_x = torch.stack((pos_x[ :, 0::2].sin(), pos_x[ :, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[ :, 0::2].sin(), pos_y[ :, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[ :, 0::2].sin(), pos_z[ :, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_x, pos_y, pos_z), dim=1).contiguous()
        return pos


  
def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward

class DAGCondGNNEncoder(nn.Module):
  """Configurable Reward-conditioned GNN Encoder for DAG task
      -takes reward(scalar) as extra input
      -conditioned on problem graph including nodes and edges
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               guidance = 0, *args, **kwargs):
    super().__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.classifier_free_guidance = guidance
    
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    
    #add reward embedding dim
    reward_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.slu_edge_embed = nn.Linear(hidden_dim, hidden_dim)
    
    if not node_feature_only:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    
    #add reward embedding
    self.reward_embed = nn.Sequential(
        linear(hidden_dim, reward_embed_dim),
        nn.ReLU(),
        linear(reward_embed_dim, reward_embed_dim),
    )
    
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    #add reward embed layers
    self.reward_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                reward_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])


    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  #def dense_forward(self, x, pbm_graph, slu_graph, timesteps, rewards):
  def dense_forward(self, x, slu_graph, timesteps, rewards):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        rewards: Input rewards (B)
    Returns:
        Updated edge features (B x V x V)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x))
    #e_pbm= self.pbm_edge_embed(self.edge_pos_embed(pbm_graph))
    e_slu = self.slu_edge_embed(self.edge_pos_embed(slu_graph))
    e = e_slu
    
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    
    if self.classifier_free_guidance:
      rwd_mask = rewards[:, 1].view(-1,1)
      rewards = rewards[:, 0]
      reward_emb = self.reward_embed(reward_embedding(rewards, self.hidden_dim))
      reward_emb = reward_emb * (1 - rwd_mask)
    else:
      rewards = rewards.view(-1)
      reward_emb = self.reward_embed(reward_embedding(rewards, self.hidden_dim))
   
    graph = torch.ones_like(slu_graph).long()

    for layer, time_layer, reward_layer, out_layer in zip(self.layers, self.time_embed_layers, self.reward_embed_layers, self.per_layer_out):
      #import pdb; pdb.set_trace()
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      x, e = layer(x, e, graph, mode="direct")
      if not self.node_feature_only:
        e = e + time_layer(time_emb)[:, None, None, :] + reward_layer(reward_emb)[:, None, None, :]
      else:
        x = x + time_layer(time_emb)[:, None, :] + reward_layer(reward_emb)[:, None, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
    
    e = self.out(e.permute((0, 3, 1, 2)))
    return e

  # def sparse_forward(self, x, graph, timesteps, edge_index):
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))
    return e

  # def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  # def sparse_encoding(self, x, e, edge_index, time_emb):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        single_time_emb = time_emb[:1]

        run_sparse_layer_fn = functools.partial(
            run_sparse_layer,
            add_time_on_edge=not self.node_feature_only
        )

        out = activation_checkpoint.checkpoint(
            run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
            x_in, e_in, single_time_emb
        )
        x = out[0]
        e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e = e + time_layer(time_emb)
        else:
          x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e

  # def forward(self, x, pbm_graph, timesteps, rewards, slu_graph):
  #   if self.node_feature_only:
  #     if self.sparse:
  #       return NotImplementedError
  #     else:
  #       raise NotImplementedError
  #   else:
  #     if self.sparse:
  #       return NotImplementedError
  #     else:
  #       return self.dense_forward(x, pbm_graph, slu_graph, timesteps, rewards)

  def forward(self, x, timesteps, rewards, slu_graph):
    if self.node_feature_only:
      if self.sparse:
        return NotImplementedError
      else:
        raise NotImplementedError
    else:
      if self.sparse:
        return NotImplementedError
      else:
        return self.dense_forward(x, slu_graph, timesteps, rewards)


class CondGNNEncoder(nn.Module):
  """Configurable Reward-conditioned GNN Encoder, whick takes reward(scalar) as extra input
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               separate_rwd_emb=False, XE_rwd_cond = False, guidance = 0, *args, **kwargs):
    super().__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.separate_rwd_emb = separate_rwd_emb
    self.XE_rwd_cond = XE_rwd_cond
    self.classifier_free_guidance = guidance
    
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    #add reward embedding dim
    reward_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)
    
    if node_feature_only:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    else:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
      
    
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    
    #add reward embedding
    self.reward_embed = nn.Sequential(
        linear(hidden_dim, reward_embed_dim),
        nn.ReLU(),
        linear(reward_embed_dim, reward_embed_dim),
    )
    
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    #add reward embed layers
    if not self.separate_rwd_emb:
      self.reward_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                reward_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])
      # self.reward_embed_out = nn.Sequential(
      #       nn.ReLU(),
      #       linear(
      #           reward_embed_dim,
      #           hidden_dim,
      #       )
      # )
    else:  
      self.reward_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(reward_embed_dim,hidden_dim,),
        ) 
        ])

      for _ in range(n_layers-1):
        self.reward_embed_layers.append(
          nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        )

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self, x, graph, timesteps, rewards, edge_index=None):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        rewards: Input rewards (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Updated edge features (B x V x V)
    """
    # Embed edge features
    del edge_index
    import pdb; pdb.set_trace()
    x = self.node_embed(self.pos_embed(x))
    e = self.edge_embed(self.edge_pos_embed(graph))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    if self.classifier_free_guidance:
      rwd_mask = rewards[:, 1].view(-1,1)
      rewards = rewards[:, 0]
      reward_emb = self.reward_embed(reward_embedding(rewards, self.hidden_dim))
      reward_emb = reward_emb * (1 - rwd_mask)
    else:
      rewards = rewards.view(-1)
      reward_emb = self.reward_embed(reward_embedding(rewards, self.hidden_dim))
    
    graph = torch.ones_like(graph).long()

    for layer, time_layer, reward_layer, out_layer in zip(self.layers, self.time_embed_layers, self.reward_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError
      
      x, e = layer(x, e, graph, mode="direct")
      if self.node_feature_only:
        x = x + time_layer(time_emb)[:, None, :] + reward_layer(reward_emb)[:, None, :]
      else:
        e = e + time_layer(time_emb)[:, None, None, :] + reward_layer(reward_emb)[:, None, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
    e = self.out(e.permute((0, 3, 1, 2)))
    return e

  def sparse_forward(self, x, graph, timesteps, rewards, edge_index):
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        rewards: Input reward features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    reward_emb = self.reward_embed(rewards)
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb, reward_emb)
    e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))
    return e

  def dense_forward_node_feature_only(self, x, graph, timesteps, rewards, edge_index=None):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        rewards: Input rewards (B)
        edge_index: None
    Returns:
        Updated node features (B x V x 2)
    """
    # Embed edge features
    del edge_index
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = self.edge_embed(self.edge_pos_embed(graph))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    reward_emb = self.reward_embed(rewards)
    graph = torch.ones_like(graph).long()

    for layer, time_layer, reward_layer, out_layer in zip(self.layers, self.time_embed_layers, self.reward_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError
    
      x, e = layer(x, e, graph, mode="direct")
      #e_inter = e * reward_layer(reward_emb)[:, None, None, :]
      #x_inter = x * reward_layer(reward_emb)[:, None, :]
      #e = e + time_layer(time_emb)[:, None, None, :] + reward_layer(reward_emb)[:, None, None, :] + e_inter
      #x = x + time_layer(time_emb)[:, None, :] + reward_layer(reward_emb)[:, None, :] + x_inter

      e = e + time_layer(time_emb)[:, None, None, :] + reward_layer(reward_emb)[:, None, None, :]
      x = x + time_layer(time_emb)[:, None, :] + reward_layer(reward_emb)[:, None, :]
      
      x = x_in + x
      e = e_in + out_layer(e)
  
    #x = x + self.reward_embed_out(reward_emb)[:, None, :]
    x = self.out(x.permute((0, 2, 1)).unsqueeze(-1))
    x = x.squeeze(-1)
    return x
  
  def sparse_forward_node_feature_only(self, x, timesteps, rewards, edge_index, node_count, edge_count):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    if self.classifier_free_guidance:
      rwd_mask = rewards[:, 1].view(-1,1)
      rewards = rewards[:, 0]
      reward_emb = self.reward_embed(reward_embedding(rewards, self.hidden_dim))
      reward_emb = reward_emb * (1 - rwd_mask)
    else:
      rewards = rewards.view(-1)
      reward_emb = self.reward_embed(reward_embedding(rewards, self.hidden_dim))
      #reward_emb = self.reward_embed(rewards)
    edge_index = edge_index.long()
    
    x, e = self.sparse_encoding(x, e, edge_index, time_emb, reward_emb, node_count=node_count, edge_count=edge_count)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb, reward_emb=None, node_count=None, edge_count=None):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, reward_layer, out_layer in zip(self.layers, self.time_embed_layers, self.reward_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)

        #apply time and reward conditioning
        if self.XE_rwd_cond == 'E':
          e = e + reward_layer(reward_emb) + time_layer(time_emb)
        elif self.XE_rwd_cond == 'X':
          x = x + reward_layer(reward_emb) + time_layer(time_emb)
        elif self.XE_rwd_cond == 'XE':
          reward_output = reward_layer(reward_emb)
          time_output = time_layer(time_emb)
          x = x + reward_output + time_output
          
          if not time_output.shape[0]==1:
            bs = time_output.shape[0] // node_count
            picked_idx = [i * node_count for i in range(bs)]
            
            reward_output = reward_output[picked_idx , :]
            reward_output = reward_output.repeat_interleave(edge_count, dim=0)

            time_output = time_output[picked_idx , :]
            time_output = time_output.repeat_interleave(edge_count, dim=0)
          
          e = e + reward_output + time_output if reward_emb is not None else e + time_output
          
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e

  def forward(self, x, timesteps, rewards, graph=None, edge_index=None, node_count=None, edge_count=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, timesteps, rewards, edge_index, node_count, edge_count)
      else:
        self.pos_embed = PositionEmbeddingSine(self.hidden_dim // 2, normalize=True)
        self.edge_pos_embed = ScalarEmbeddingSine(self.hidden_dim, normalize=False)  
        return self.dense_forward_node_feature_only(x, graph, timesteps, rewards, edge_index)
    else:
      if self.sparse:
        return self.sparse_forward(x, graph, timesteps, rewards, edge_index)
      else:
        return self.dense_forward(x, graph, timesteps, rewards, edge_index)


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               sparse_noise = False,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.sparse_noise = sparse_noise
    
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    if not node_feature_only:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self, x, graph, timesteps, edge_index=None):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Updated edge features (B x V x V)
    """
    # Embed edge features
    del edge_index
    
    x = self.node_embed(self.pos_embed(x))
    e = self.edge_embed(self.edge_pos_embed(graph))
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    graph = torch.ones_like(graph).long()

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      x, e = layer(x, e, graph, mode="direct")
      if not self.node_feature_only:
        e = e + time_layer(time_emb)[:, None, None, :]
      else:
        x = x + time_layer(time_emb)[:, None, :]
      x = x_in + x
      e = e_in + out_layer(e)
    e = self.out(e.permute((0, 3, 1, 2)))
    return e

  def sparse_forward(self, x, graph, timesteps, edge_index):
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    #e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2)
    e = e.reshape((1, 1, -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))

    return e
  
  def sparse_noise_forward(self, x, graph, timesteps, edge_index, problem_size):
    # use the outproduct of the node features for edge predictions
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    
    x = x.reshape(-1, problem_size, self.hidden_dim)
    x= x.unsqueeze(2)
    x= x.repeat(1, 1, x.size(1), 1)
    edge_pred = x + x.transpose(1, 2)
    edge_pred = self.out(edge_pred.permute(0,3,1,2)).reshape(1, 2, -1)
    
    return edge_pred

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        pass
        # single_time_emb = time_emb[:1]

        # run_sparse_layer_fn = functools.partial(
        #     run_sparse_layer,
        #     add_time_on_edge=not self.node_feature_only
        # )

        # out = activation_checkpoint.checkpoint(
        #     run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
        #     x_in, e_in, single_time_emb
        # )
        # x = out[0]
        # e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e = e + time_layer(time_emb)
        else:
          x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e

  def forward(self, x, timesteps, graph=None, edge_index=None, sparse_inf=False, problem_size=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
      else:
        raise NotImplementedError
    else:
      if self.sparse and not sparse_inf:
        # if self.sparse_noise:
        #   return self.sparse_noise_forward(x, graph, timesteps, edge_index, problem_size)
        # else:
        return self.sparse_forward(x, graph, timesteps, edge_index)
      else:
        return self.dense_forward(x, graph, timesteps, edge_index)
