import numpy as np

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from torch.nn import Parameter, Linear, LayerNorm, ReLU
from torch_geometric.nn.inits import glorot, zeros, reset
from torch_geometric.utils import softmax, dropout_adj
from torch_sparse import coalesce, spmm
from torch_scatter import scatter

class MLP(nn.Sequential):
    # Bias and dropout is not changed like this in the DeepGCN paper.
    def __init__(self, channels, dropout, bias):
        m = []
        for i in range(1, len(channels)):
            # Hidden Layer
            if i < len(channels) - 1:
                m.append(LayerNorm(channels[i - 1], elementwise_affine=True))
                m.append(Linear(channels[i - 1], channels[i], bias=bias))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))
            # Output Layer
            else:
                m.append(Linear(channels[i - 1], channels[i], bias=bias))
        super(MLP, self).__init__(*m)

class PPRPowerIteration(nn.Module):
    def __init__(self, dim, heads, pow_iter, alpha, att_dropout):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.pow_iter = pow_iter
        self.alpha = alpha
        self.att_dropout = att_dropout

    def forward(self, edge_index, edge_weight, entity_embed):
        """Compute Attention Diffusion Matrix for every attention head based on the Power Iteration Equation"""

        # Sparse Transition Matrix for every attention head
        # [n_triples, heads]
        A = (1 - self.alpha) * edge_weight

        # Compute Attention Diffusion Matrix for every attention head
        head_outs = []
        for head in range(self.heads):

            # [n_entities, in_channels]
            Z_0 = entity_embed
            Z_K = entity_embed
            n_entities = entity_embed.shape[0]

            # [n_triples]
            A_i = A[:, head]

            # Attention Dropout: [n_triples]
            A_i_drop = F.dropout(A_i, p=self.att_dropout, training=self.training)

            for _ in range(self.pow_iter):
                # Attention Dropout: [n_triples]
                # A_i_drop = F.dropout(A_i, p=self.att_dropout, training=self.training)
                # [n_entities, in_channels] = [n_entities, n_entities] * [n_entities, in_channels] + [n_entities, in_channels]
                Z_K = spmm(edge_index, A_i_drop, n_entities, n_entities, Z_K) + self.alpha * Z_0

            # Cache Attention Diffusion Matrix for every attention head
            # [n_entities, 1, in_channels]
            head_outs.append(Z_K.unsqueeze(1))

        # Stack Attention Diffusion Matrices
        # [n_entities, heads, in_channels]
        out = torch.cat(head_outs, dim = 1)

        return out

class DAGNConv(nn.Module):
    def __init__(self, dim, heads, pow_iter, alpha, att_dropout, bias=False):
        super().__init__()

        #======================General Layer Architecture=======================
        # in_channel/out_channel dimension
        self.dim = dim
        # number of attention heads to use
        self.heads = heads
        # number of power iterations
        self.pow_iter = pow_iter
        # PPR teleport probability
        self.alpha = alpha  
        # message / attention dropout rate 
        self.att_dropout = att_dropout
        # learn an additional bias per Linear layer or not
        self.bias = bias
        # initialize attention diffusion mechanism
        self.propagation = PPRPowerIteration(self.dim, self.heads, self.pow_iter, self.alpha, self.att_dropout)

        # maybe include later the following options (But we can simply also decide to use the defaul values 'False', 0.2)
        # add self-loops to the adjacency-matrix or not (NOTE: DON'T INCLUDE AS WE HAVE A KNOWLEDGE GRAPH!!!)
        # self.add_self_loops = add_self_loops
        # choose slope for LeakyReLU
        # self.negative_slope = negative_slope

        #===========================Initialize Weights======================
        # Projection Matrices: [in_channels, heads * out_channels]
        # Maybe a sinle matrix is also sufficient here, just as done in GAT
        self.W_h = nn.Linear(self.dim, self.heads * self.dim, bias=False)
        self.W_t = nn.Linear(self.dim, self.heads * self.dim, bias=False)
        self.W_r = nn.Linear(self.dim, self.heads * self.dim, bias=False)

        # Attention Mechanisms [1, heads, out_channels]
        # Paper: [Wh || Wt || Wr]-concatenation and dot product with attention mechanism "v" with shape [1, heads, 3 * out_channels]
        # Here: Built dot product between Wh and "att_h", Wt and "att_t", Wr and "att_r" respecively and built the sum over them for faster computation
        self.att_h = nn.Parameter(torch.Tensor(1, self.heads, self.dim))
        self.att_t = nn.Parameter(torch.Tensor(1, self.heads, self.dim))
        self.att_r = nn.Parameter(torch.Tensor(1, self.heads, self.dim))

        # Projection Matrices: [heads * in_channels, out_channels]
        self.W_o = nn.Linear(self.heads * self.dim, self.dim, bias=False)

        # Bias is not crucial to DAGN - feel free to experiment
        if self.bias:
            # [heads * out_channels]
            self.bias_h = nn.Parameter(torch.Tensor(self.heads * self.dim))
            self.bias_t = nn.Parameter(torch.Tensor(self.heads * self.dim))
            self.bias_r = nn.Parameter(torch.Tensor(self.heads * self.dim))
            # [out_channels]
            self.bias_o = nn.Parameter(torch.Tensor(self.dim))
        else:
            # [None]
            self.register_parameter('bias_h', None)
            self.register_parameter('bias_t', None)
            self.register_parameter('bias_r', None)
            self.register_parameter('bias_o', None)

    def reset_parameters(self):
        # Use glorot (aka. xavier uniform initialization)
        glorot(self.W_h.weight)
        glorot(self.W_t.weight)
        glorot(self.W_r.weight)
        glorot(self.W_o.weight)
        glorot(self.att_h)
        glorot(self.att_t)
        glorot(self.att_r)
        zeros(self.bias_h)
        zeros(self.bias_t)
        zeros(self.bias_r)
        zeros(self.bias_o)

    def forward(self, entity_embed, relation_embed, edge_index, edge_type):
        # edge_index doesn't have the shape [2, n_triples]
        assert edge_index.shape[0] == 2
        # edge_type doesn't have the shape [n_triples]
        assert len(edge_type.shape) == 1
        # edge_index and edge_type don't have the same [n_triples]-length
        assert edge_index.shape[1] == edge_type.shape[0]

        head_ids, tail_ids = edge_index[0], edge_index[1]
        relation_ids = edge_type

        #==========================Eq. 1/2: Edge Attention Computation==========================
        # get entity and relation embeddings 
        # note: using index_select is faster than indexing (entity_embed[head_ids]) in PyTorch!
        # [n_triples, in_channels]
        e_h = entity_embed.index_select(0, head_ids)
        e_t = entity_embed.index_select(0, tail_ids)
        e_r = relation_embed.index_select(0, relation_ids)

        # perform linear projection
        # [n_triples, heads * out_channels] = [n_triples, in_channels] * [in_channels, heads * out_channels] 
        # Using view [n_triples, heads * out_channels] -> [n_triples, heads, out_channels]
        Wh = self.W_h(e_h).view(-1, self.heads, self.dim)
        Wt = self.W_t(e_t).view(-1, self.heads, self.dim)
        Wr = self.W_r(e_r).view(-1, self.heads, self.dim)

        if self.bias:
            Wh += self.bias_h
            Wt += self.bias_t
            We += self.bias_r

        # attention mechanism
        # [n_triples, heads] = torch.sum([n_triples, heads, out_channels] * [1, heads, out_channels], keepdims=False)
        # BEMERKE: WENN ICH AN DIESER STELLE keepdims=True mache, muss ich die attention unten nicht .unsqueeze(-1) machen
        att_scores_h = torch.sum((torch.tanh(Wh) * self.att_h), dim=-1)
        att_scores_t = torch.sum((torch.tanh(Wt) * self.att_t), dim=-1)
        att_scores_r = torch.sum((torch.tanh(Wr) * self.att_r), dim=-1)

        # [n_triples, heads]
        att_scores = F.leaky_relu(att_scores_h + att_scores_t + att_scores_r)
        # [n_triples, heads]
        att_scores_normalized = softmax(att_scores, head_ids)

        #==========================Eq.3: Attention Diffusion==========================
        # Equation 3: Attention Diffusion
        # [n_entities, heads, in_channels]
        out = self.propagation(edge_index, att_scores_normalized, entity_embed)
        # Concat Attention Diffusion Matrices: [n_entities, heads * in_channels]
        out = out.view(-1, self.heads * self.dim)
        # [n_entities, out_channels] = [n_entities, heads * in_channels] * [heads * in_channels, out_channels]
        out = self.W_o(out)
      
        if self.bias:
            out += self._build_model

        return out

class DAGNLayer(torch.nn.Module):
    def __init__(self, norm=None, conv=None, mlp=None, feat_dropout=0.,ckpt_grad=False):
        super(DAGNLayer, self).__init__()

        self.norm = norm
        self.conv = conv
        self.mlp = mlp
        self.feat_dropout = feat_dropout
        self.ckpt_grad = ckpt_grad

        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()
        reset(self.mlp)

    def forward(self, entity_embed, relation_embed, edge_index, edge_type):

        #===============Eq.6: Multi-Head Attention Diffusion Layer===============
        # Cache Residual
        x = entity_embed  
        # Layer Norm
        entity_embed = self.norm(x) 
        # Graph Diffusion Convolution
        entity_embed = F.dropout(entity_embed, p=self.feat_dropout, training=self.training)
        if self.ckpt_grad:
            entity_embed = checkpoint(self.conv, entity_embed, relation_embed, edge_index, edge_type)
        else:
            entity_embed = self.conv(entity_embed, relation_embed, edge_index, edge_type)
        # Residual Connection
        entity_embed = entity_embed + x

        #======================Eq.7: Deep Aggregation Layer======================
        # Cache Residual
        x = entity_embed
        # MLP
        entitiy_embed = self.mlp(entity_embed)
        # Residual Connection
        entitiy_embed = entitiy_embed + x

        return entity_embed

class DAGNLinkPrediction(torch.nn.Module):
    def __init__(self, args, n_entities, n_relations, edge_index, edge_type):
        super().__init__()
        self._parse_args(args, n_entities, n_relations, edge_index, edge_type)
        self._build_weights()
        self._build_model()

    def _parse_args(self, args, n_entities, n_relations, edge_index, edge_type):

        self.edge_index = edge_index
        self.edge_type = edge_type

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.n_layers = args.n_layers
        self.heads = args.heads
        self.dim = args.dim 
        self.hidden_dim = args.hidden_dim
        self.pow_iter = args.pow_iter
        self.alpha = args.alpha
        self.att_dropout = args.att_dropout
        self.feat_dropout = args.feat_dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_weights(self):

        initializer = nn.init.xavier_uniform_
        self.entity_embed = nn.Parameter(initializer(torch.empty(self.n_entities, self.dim)))
        self.relation_embed = nn.Parameter(initializer(torch.empty(self.n_relations, self.dim)))

    def _build_model(self):

        self.layers = torch.nn.ModuleList()

        # 1st layer, ... , n-th layer
        for layer in range(self.n_layers):
            norm = LayerNorm(self.dim, elementwise_affine=True)
            conv = DAGNConv(self.dim, self.heads, self.pow_iter, self.alpha, self.att_dropout, bias=False) 
            mlp = MLP([self.dim, self.hidden_dim, self.dim], dropout=self.feat_dropout, bias=False)
            layer = DAGNLayer(norm, conv, mlp, self.feat_dropout, ckpt_grad=False) # layer%2
            self.layers.append(layer)
    
    def encode(self):

        # Get embeddings
        entity_embed, relation_embed = self.entity_embed, self.relation_embed
        # Get adjacency matrix
        edge_index, edge_type = self.edge_index, self.edge_type

        # Node Dropout: Is excluding triples from graph, thus edge_index, edge_type size is decreaced
        # edge_index, edge_type = dropout_adj(edge_index, edge_type, p = self.node_dropout, training=self.training)
        for layer in self.layers:
            entity_embed = layer(entity_embed, relation_embed, edge_index, edge_type)

        return entity_embed, relation_embed

    def decode(self, args, entity_embed, relation_embed, batch):
       # See also https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/link_predict.py

        # Get indices of current batch
        head_indices = batch[:, 0]
        relation_indices = batch[:, 1]
        #tail_indeices = batch[:, 2]

        # Get embeddings for current batch
        # [batch_size, dim]
        s = entity_embed.index_select(0, head_indices)
        r = relation_embed.index_select(0, relation_indices) #Note that relation_embed is eq to M_r in the original paper
        #o = entity_embed.index_select(0, tail_indices)

        # DistMult Decoder
        # [batch_size, n_entities] = ([batch_size, dim] * [batch_size, dim]) * [n_entities, dim].T
        scores = torch.mm(s*r, entity_embed.transpose(1,0))
        #scores = torch.sum(s*r*o, dim=1)

        if args.loss == 'BCE':
            preds = torch.sigmoid(scores)
        else:
            preds = F.log_softmax(scores, dim=1)

        return preds