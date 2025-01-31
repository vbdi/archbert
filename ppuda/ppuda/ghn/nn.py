# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Graph HyperNetworks.

"""

import torch
import torch.nn as nn
import os
from .mlp import MLP
#from .decoder import MLPDecoder, ConvDecoder
from .layers import ShapeEncoder
from ..deepnets1m.ops import NormLayers, PosEnc
from ..deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
from ..deepnets1m.net import named_layered_modules
from ..deepnets1m.graph import Graph, GraphBatch
from ..utils import capacity, default_device, adjust_net
import time
#import torchvision.models as models
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    https://github.com/Diego999/pyGAT/blob/master/layers.py
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, device='cpu'):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.device=device

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features), device=device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1), device=device))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        """
        :param h: (batch_zize, number_nodes, in_features)
        :param adj: (batch_size, number_nodes, number_nodes)
        :return: (batch_zize, number_nodes, out_features)
        """
        # batchwise matrix multiplication
        Wh = torch.matmul(h, self.W)  # (batch_zize, number_nodes, in_features) * (in_features, out_features) -> (batch_zize, number_nodes, out_features)
        e = self.prepare_batch(Wh)  # (batch_zize, number_nodes, number_nodes)

        # (batch_zize, number_nodes, number_nodes)
        zero_vec = -9e15 * torch.ones_like(e).to(self.device)

        # (batch_zize, number_nodes, number_nodes)
        attention = torch.where(adj > 0, e, zero_vec)

        # (batch_zize, number_nodes, number_nodes)
        attention = F.softmax(attention, dim=-1)

        # (batch_zize, number_nodes, number_nodes)
        #attention = F.dropout(attention, self.dropout, training=self.training)

        # batched matrix multiplication (batch_zize, number_nodes, out_features)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def prepare_batch(self, Wh):
        """
        with batch training
        :param Wh: (batch_zize, number_nodes, out_features)
        :return:
        """
        # Wh.shape (B, N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (B, N, 1)
        # e.shape (B, N, N)

        B, N, E = Wh.shape  # (B, N, N)

        # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (B, N, out_feature) X (out_feature, 1) -> (B, N, 1)

        # broadcast add (B, N, 1) + (B, 1, N)
        e = Wh1 + Wh2.permute(0, 2, 1)  # (B, N, N)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads, device='cpu'):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = nn.ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, device=device) for _ in range(nheads)])
        #for i, attention in enumerate(self.attentions):
        #    self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False, device=device)

    def forward(self, x, adj):
        #x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        # for classification (we dont need it)
        #return F.log_softmax(x, dim=1)
        return x

class GHN(nn.Module):
    r"""
    Graph HyperNetwork based on "Chris Zhang, Mengye Ren, Raquel Urtasun. Graph HyperNetworks for Neural Architecture Search. ICLR 2019."
    (https://arxiv.org/abs/1810.05749)
    """

    # ni
    def __init__(self,
                 max_shape=(64, 64, 11, 11),
                 hypernet='gat',
                 ve=False,
                 layernorm=False,
                 hid=768,
                 debug_level=0,
                 device='cpu'):
        super(GHN, self).__init__()

        assert len(max_shape) == 4, max_shape
        self.layernorm = layernorm
        self.ve = ve
        self.debug_level = debug_level

        if layernorm:
            self.ln = nn.LayerNorm(hid)

        self.device = device
        self.embed = torch.nn.Embedding(len(PRIMITIVES_DEEPNETS1M), hid, device=device)
        self.shape_enc = ShapeEncoder(hid=hid, max_shape=max_shape, debug_level=debug_level, device=device)

        if hypernet == 'mlp':
            self.gnn = MLP(in_features=hid, hid=(hid, hid), device=device)
        elif hypernet=='gat':
            #self.gnn = GraphAttentionLayer(hid, hid, 0.6, 0.2).to(device)
            self.gnn = GAT(hid, hid, 0.6, 0.2, 4, device=device)
        else:
            self.gnn = None
            #raise NotImplementedError(hypernet)


    @staticmethod
    def load(checkpoint_path, debug_level=1, device=default_device(), verbose=False):
        state_dict = torch.load(checkpoint_path, map_location=device)
        ghn = GHN(**state_dict['config'], debug_level=debug_level).to(device).eval()
        ghn.load_state_dict(state_dict['state_dict'])
        #if verbose:
        #    print('GHN with {} parameters loaded from epoch {}.'.format(capacity(ghn)[1], state_dict['epoch']))
        return ghn

    # ni
    def pooling_forward(self, token_embeddings):
        #token_embeddings = features['token_embeddings']
        #token_embeddings = token_embeddings.unsqueeze(0)
        attention_mask = torch.ones(token_embeddings.shape)#features['attention_mask']

        ## Pooling strategy
        output_vectors = []
        '''
        if self.pooling_mode_cls_token:
            cls_token = features.get('cls_token_embeddings', token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)            
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
        '''
        input_mask_expanded = attention_mask.expand(token_embeddings.size()).float().to(self.device)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
        #if 'token_weights_sum' in features:
        #    sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
        #else:
        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)

        #if self.pooling_mode_mean_tokens:
        output_vectors.append(sum_embeddings / sum_mask)
        #if self.pooling_mode_mean_sqrt_len_tokens:
        #    output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))

        output_vector = torch.cat(output_vectors, 1)
        #features.update({'sentence_embedding': output_vector})
        return output_vector

    # ni
    def forward(self, node_feats, shape_inds, edges=None, adjs=None, return_embeddings=False, predict_class_layers=True, bn_train=True):
        r"""
        Predict parameters for a list of >=1 networks.
        :param nets_torch: one network or a list of networks, each is based on nn.Module.
                           In case of evaluation, only one network can be passed.
        :param graphs: GraphBatch object in case of training.
                       For evaluation, graphs can be None and will be constructed on the fly given the nets_torch in this case.
        :param return_embeddings: True to return the node embeddings obtained after the last graph propagation step.
                                  return_embeddings=True is used for property prediction experiments.
        :param predict_class_layers: default=True predicts all parameters including the classification layers.
                                     predict_class_layers=False is used in fine-tuning experiments.
        :param bn_train: default=True sets BN layers in nets_torch into the training mode (required to evaluate predicted parameters)
                        bn_train=False is used in fine-tuning experiments
        :return: nets_torch with predicted parameters and node embeddings if return_embeddings=True
        """

        '''
        if not self.training:
            assert isinstance(nets_torch,
                              nn.Module) or len(nets_torch) == 1, \
                'constructing the graph on the fly is only supported for a single network'

            if isinstance(nets_torch, list):
                nets_torch = nets_torch[0]

            if self.debug_level:
                if self.debug_level > 1:
                    valid_ops = graphs[0].num_valid_nodes(nets_torch)
                start_time = time.time()  # do not count any debugging steps above

            if graphs is None:
                graphs = GraphBatch([Graph(nets_torch, ve_cutoff=50 if self.ve else 1)])
                graphs.to_device(self.embed.weight.device)

        else:
            assert graphs is not None, \
                'constructing the graph on the fly is only supported in the evaluation mode'

        '''

        '''
        if nets_torch is None:
            nets_torch = [adjust_net(models.resnet18(num_classes=10)).eval(),
                          adjust_net(models.resnet50(num_classes=10)).eval(),]
                          #adjust_net(models.resnet18(num_classes=10)).eval(),
                          #adjust_net(models.resnet50(num_classes=10)).eval()]
        

        # ni process over every single model and graph
        # Find mapping between embeddings and network parameters
        param_groups = []
        params_map = []
        #for (graph, net_torch) in zip(graphs, nets_torch):
        #for net_torch in nets_torch:
        if graphs:
            if isinstance(graphs, GraphBatch):
                graph = graphs
            else:
                graph = GraphBatch(graphs, padding=self.graph_padding)
        else:
            graph_list = []
            if type(nets_torch) == list:
                for i in range(len(nets_torch)):
                    g = Graph(adjust_net(nets_torch[i], large_input=False), ve_cutoff=50 if self.ve else 1)
                    graph_list.append(g)
            else:
                graph_list.append(Graph(adjust_net(nets_torch, large_input=False), ve_cutoff=50 if self.ve else 1))
            graph = GraphBatch(graph_list, padding=self.graph_padding)
        graph.to_device(self.embed.weight.device)
        '''
        #param_groups, params_map = self._map_net_params(graph, nets_torch, sanity_check = self.debug_level > 0)
        #param_groups.append(param_group)
        #params_map.append(param_map)
        '''
        if self.debug_level or not self.training:
            n_params_true = sum([capacity(net)[1] for net in (nets_torch if isinstance(nets_torch, list) else [nets_torch])])
            if self.debug_level > 1:
                print('\nnumber of learnable parameter tensors: {}, total number of parameters: {}'.format(
                    valid_ops, n_params_true))
        '''

        # ni trainable items start from this point.
        ### padding is required for graph.node_feat[:, 0], graph.node_feat[:, 1], graph.edges
        ### after padding, create the batch for them all
        ### then give it to initial, spatial, channel embedders, and also GNN

        # Obtain initial embeddings for all nodes # batching is fine here!
        init_embed = self.embed(node_feats)
        # ni if dimensions not the same - we can use padding! # batching is fine here!
        x = self.shape_enc(init_embed, shape_inds)

        # gat
        if self.gnn:
            x = self.gnn(x, adjs)

        # Update node embeddings using a GatedGNN, MLP or another model
        # ni graph.node_feat[:, 1] is always 0 for batch size of 1
        # NOTE: batching is NOT fine here :(
        #x = self.gnn(x, graph.edges, graph.node_feat[:, 1])

        if self.layernorm:
            x = self.ln(x)

        #import sentence_transformers
        #pooling_model = sentence_transformers.models.Pooling(768)
        x_pooling = self.pooling_forward(x)

        '''
        # ni let's create the batches
        #first_item = x[]
        start_ind = 0
        x_pooling = self.pooling_forward(x[start_ind:graph.n_nodes[0]])
        if self.graph_padding:
            x_batched = x[start_ind:graph.n_nodes[0]].unsqueeze(dim=0)
        else:
            x_batched = None
        for n in graph.n_nodes[1:]:
            # ni pooling is required (similar to sentence-transformers) to get a fixed size embed for arch
            x_pooling = torch.cat([x_pooling, self.pooling_forward(x[start_ind:start_ind+n])])
            if self.graph_padding:
                x_batched = torch.concat([x_batched, x[start_ind:start_ind + n].unsqueeze(dim=0)])
            start_ind = start_ind + n
        '''

        # ni this is the last point embedding the architecture

        #return (nets_torch, x) if return_embeddings else nets_torch
        return x, x_pooling

    def _map_net_params_v2(graph, sanity_check=False):
        r"""
        Matches the parameters in the models with the nodes in the graph.
        Performs additional steps.
        :param graphs: GraphBatch object
        :param nets_torch: a single neural network of a list
        :param sanity_check:
        :return: mapping, params_map
        """
        #params_maps_list = []
        #nets_torch = [nets_torch] if type(nets_torch) not in [tuple, list] else nets_torch

        #for b, (node_info, net) in enumerate(zip(graphs.node_info, nets_torch)):
        #for b, node_info in enumerate(graphs.node_info):
        node_info = graph.node_info
        mapping = {}
        params_map = {}

        net = graph.model
        target_modules = named_layered_modules(net)

        param_ind = 0#torch.sum(graph.n_nodes).item()

        #param_ind = torch.sum(graphs.n_nodes[:b]).item()

        for cell_id in range(len(node_info)):
            matched_names = []
            for (node_ind, param_name, name, sz, last_weight, last_bias) in node_info[cell_id]:
                matched = []
                for m in target_modules[cell_id]:
                    if m['param_name'].startswith(param_name):
                        #### ni if the arch-only is being sent here
                        if last_bias:
                            if hasattr(m['module'], 'bias'):
                                m['sz'] = torch.Size(m['module'].bias.data.clone().detach().int())
                            else:
                                m['sz'] = torch.Size(m['module'].weight.data.clone().detach().int())
                        else:
                            if len(m['module'].weight.shape)>1: # special case for conv1d
                                m['sz'] = torch.Size(m['module'].weight_v.data.clone().detach().int())
                            else:
                                m['sz'] = torch.Size(m['module'].weight.data.clone().detach().int())
                        #########
                        matched.append(m)
                        if not sanity_check:
                            break
                if len(matched) > 1:
                    raise ValueError(cell_id, node_ind, param_name, name, [
                        (t, (m.weight if is_w else m.bias).shape) for
                        t, m, is_w in matched])
                elif len(matched) == 0:
                    if sz is not None:
                        params_map[param_ind + node_ind] = ({'sz': sz}, None, None)

                    if sanity_check:
                        for pattern in ['input', 'sum', 'concat', 'pool', 'glob_avg', 'msa', 'cse']:
                            good = name.find(pattern) >= 0
                            if good:
                                break
                        assert good, \
                            (cell_id, param_name, name,
                             node_info[cell_id],
                             target_modules[cell_id])
                else:
                    matched_names.append(matched[0]['param_name'])

                    sz = matched[0]['sz']

                    if len(sz) == 1:
                        key = 'cls_b' if last_bias else '1d'
                    elif last_weight:
                        key = 'cls_w'
                    else:
                        if len(sz) == 3:
                            key = '4d-%d-%d' % (sz[1:])
                        else:
                            key = '4d-%d-%d' % ((1, 1) if len(sz) == 2 else sz[2:])
                    if key not in mapping:
                        mapping[key] = []
                    params_map[param_ind + node_ind] = (matched[0], key, len(mapping[key]))
                    mapping[key].append(param_ind + node_ind)

            # ni not needed for now
            #assert len(matched_names) == len(set(matched_names)), (
            #    'all matched names must be unique to avoid predicting the same paramters for different moduels',
            #    len(matched_names), len(set(matched_names)))
            matched_names = set(matched_names)

            # Prune redundant ops in Network by setting their params to None
            for m in target_modules[cell_id]:
                if m['is_w'] and m['param_name'] not in matched_names:
                    m['module'].weight = None
                    if hasattr(m['module'], 'bias') and m['module'].bias is not None:
                        m['module'].bias = None

            #params_maps_list.append(params_map)

        return mapping, params_map
        #return None, params_maps_list