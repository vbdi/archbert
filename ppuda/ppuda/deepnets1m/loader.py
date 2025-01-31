# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Loaders for DeepNets-1M.

"""


import numpy as np
import torch.utils.data
#import torchvision
import json
import h5py
import os
from ..deepnets1m import genotypes
from ..utils import adjust_net, rand_choice
from .genotypes import from_dict, PRIMITIVES_DEEPNETS1M
from .ops import *
from .net import Network
from .graph import Graph, GraphBatch, graph_padding

from sentence_transformers.readers import InputExample
import time

#import sys
#sys.path.append('.')
from ppuda.ppuda.ghn.nn import GHN
from ppuda.ppuda.ghn.layers import ShapeEncoder
from ppuda.ppuda.utils import adjust_net


MAX_NODES_BATCH = 2200  # to fit larger meta batches into GPU memory (decreasing this number further may create a bias towards smaller architectures)

class DeepNets1M(torch.utils.data.Dataset):
    r"""
    Default args correspond to training a baseline GHN on CIFAR-10.
    """

    def __init__(self,
                 split='train',
                 nets_dir='./data',
                 virtual_edges=1,
                 num_ch=(32, 128),
                 fc_dim=(64, 512),
                 num_nets=None,
                 arch=None,
                 large_images=False, max_node_size=512, max_edge_size=512, dataset_type='default', mode='single'):
        super(DeepNets1M, self).__init__()

        self.max_node_size = max_node_size
        self.max_edge_size = max_edge_size

        self.dataset_type = dataset_type
        self.mode = mode

        self.split = split
        assert self.split in ['train', 'val', 'test', 'search',
                              'wide', 'deep', 'dense', 'bnfree', 'predefined'],\
            ('invalid split', self.split)
        self.is_train = self.split == 'train'

        self.virtual_edges = virtual_edges
        assert self.virtual_edges >= 1, virtual_edges

        if True:#self.is_train:
            # During training we will randomly sample values from this range
            self.num_ch = torch.arange(num_ch[0], num_ch[1] + 1, 16)
            self.fc_dim = torch.arange(fc_dim[0], fc_dim[1] + 1, 64)

        self.large_images = large_images  # this affects some network parameters

        # Load one of the splits
        print('\nloading %s nets...' % self.split.upper())

        if self.split == 'predefined':
            self.nets = self._get_predefined()
            n_all = len(self.nets)
            self.nodes = torch.tensor([net.n_nodes for net in self.nets])
        else:
            self.h5_data = None
            self.h5_file = os.path.join(nets_dir, 'autonet_%s.hdf5' % (split if split in ['train', 'search'] else 'val'))

            self.primitives_dict = {op: i for i, op in enumerate(PRIMITIVES_DEEPNETS1M)}
            assert os.path.exists(self.h5_file), ('%s not found' % self.h5_file)

            # ni for qa dataset
            # load all the unique answers (labels) for QA task
            if self.dataset_type == 'qa':
                self.all_answers = []
                if self.mode == 'single':
                    qa_answers_file = open(nets_dir + '/qa_answers.txt', 'r')
                elif self.mode == 'multi':
                    qa_answers_file = open(nets_dir + '/qa_unique_answers.txt', 'r')
                for line in qa_answers_file:
                    self.all_answers.append(line.strip())
            ####

            # load all new vocabs
            self.new_vocab = []
            vocab_file = open(nets_dir + '/vocabs.txt', 'r')
            for line in vocab_file:
                self.new_vocab.append(line.strip())
            ####

            # Load meta data to convert dataset files to graphs later in the _init_graph function
            to_int_dict = lambda d: { int(k): v for k, v in d.items() }
            with open(self.h5_file.replace('.hdf5', '_meta.json'), 'r') as f:
                meta = json.load(f)[split]
                n_all = len(meta['nets'])
                self.nets = meta['nets'][:n_all if num_nets is None else num_nets]
                self.primitives_ext =  to_int_dict(meta['meta']['primitives_ext'])
                self.op_names_net = to_int_dict(meta['meta']['unique_op_names'])
            self.h5_idx = [ arch ] if arch is not None else None
            self.nodes = torch.tensor([net['num_nodes'] for net in self.nets])

        if arch is not None:
            arch = int(arch)
            assert arch >= 0 and arch < len(self.nets), \
                'architecture with index={} is not available in the {} split with {} architectures in total'.format(arch, split, len(self.nets))
            self.nets = [self.nets[arch]]
            self.nodes = torch.tensor([self.nodes[arch]])

        print('loaded {}/{} nets with {}-{} nodes (mean\u00B1std: {:.1f}\u00B1{:.1f})'.
              format(len(self.nets),n_all,
                     self.nodes.min().item(),
                     self.nodes.max().item(),
                     self.nodes.float().mean().item(),
                     self.nodes.float().std().item()))


    @staticmethod
    def loader(meta_batch_size=1, **kwargs):
        nets = DeepNets1M(**kwargs)
        #nets.is_train=False
        loader = torch.utils.data.DataLoader(nets,
                                             batch_sampler=NetBatchSampler(nets, meta_batch_size) if nets.is_train else None,
                                             batch_size=1, # this needs to be 1 for training
                                             pin_memory=False,
                                             #collate_fn=GraphBatch,
                                             num_workers=0)# if meta_batch_size <= 1 else min(8, meta_batch_size)
        #return iter(loader) if nets.is_train else loader
        if nets.dataset_type == 'qa':
            # we need the number of all possible answers for QA fine-tuning
            return loader, nets.all_answers, nets.new_vocab
        else:
            return loader, nets.new_vocab

    def __len__(self):
        return len(self.nets)


    def __getitem__(self, idx):

        if self.split == 'predefined':
            graph = self.nets[idx]
        else:

            if self.h5_data is None:  # A separate fd is opened for each worker process
                self.h5_data = h5py.File(self.h5_file, mode='r')

            args = self.nets[idx]
            idx = self.h5_idx[idx] if self.h5_idx is not None else idx
            cell, n_cells = from_dict(args['genotype']), args['n_cells']
            graph = self._init_graph(self.h5_data[self.split][str(idx)]['adj'][()],
                                     self.h5_data[self.split][str(idx)]['nodes'][()],
                                     n_cells)

            if True:#self.is_train:
                is_conv_dense = sum([n[0] in ['conv_5x5', 'conv_7x7'] for n in
                                     cell.normal + cell.reduce]) > 0
                num_params = args['num_params']['imagenet' if self.large_images else 'cifar10'] / 10 ** 6

                fc = rand_choice(self.fc_dim, 4)            # 64-256
                if num_params > 0.8 or not args['glob_avg'] or is_conv_dense or n_cells > 12:
                    C = self.num_ch.min()
                elif num_params > 0.4 or n_cells > 10:
                    C = rand_choice(self.num_ch, 2)         # 16-32
                elif num_params > 0.2 or n_cells > 8:
                    C = rand_choice(self.num_ch, 3)         # 16-64
                else:
                    C = rand_choice(self.num_ch)            # 16-128
                    if C <= 64:
                        fc = rand_choice(self.fc_dim)
                args['C'] = C.item()
                args['fc_dim'] = fc.item()

            net_args = {'genotype': cell}
            #auto_texts = []
            for key in ['norm', 'ks', 'preproc', 'glob_avg', 'stem_pool', 'C_mult',
                        'n_cells', 'fc_layers', 'C', 'fc_dim', 'stem_type']:
                if key == 'C' and self.split == 'wide':
                    net_args[key] = args[key] * (2 if self.large_images else 4)
                else:
                    net_args[key] = args[key]
                #auto_texts.append(str(net_args[key]) + ' ' + key)

            graph.net_args = net_args
            graph.net_idx = idx

            #t1 = time.time()
            # ni
            net = Network(is_imagenet_input=False, num_classes=10, compress_params=False, **net_args).eval()

            # ni add label description to the graph object
            if 'n_params' in args:
                graph.n_params = args['n_params']
            else:
                graph.n_params = None
            if 'n_layers' in args:
                graph.n_layers = args['n_layers']
            else:
                graph.n_layers = None
            if 'unique_layers' in args:
                graph.unique_layers = args['unique_layers']
            else:
                graph.unique_layers = None
            if 'texts' in args:
                graph.texts = args['texts']#[0:1]
            else:
                ##### put be's auto-labeller here
                # descriptions, labels = auto_label_generator(net)
                descriptions = ['some positive sample']  # , 'some negative sample']
                # Create the description on the fly (from the net_args info)
                graph.texts = descriptions  # [", ".join(auto_texts)]
            if 'label' in args:
                graph.label = args['label']  # [0:1]
            else:
                labels = [1.0]#[1.0,0.0]
                # Create the label on the fly
                # the label can be some similarity score! or we can keep it None for MultipleNegative loss, where negative samples are automatically generated.
                # some sample score [0,1] (the higher the more similar the pairs. for negative ones, we can set values close to 0.0)
                graph.label = labels
            #####

            #if 'answers' in args: # qa dataset
            if self.dataset_type == 'qa':
                ### be's code for creating one-hot vectors
                graph.label = []
                model_answers = args['answers']  # read_answers_file
                all_answers_len = len(self.all_answers)
                for answer in model_answers:
                    if self.mode == 'single':
                        if answer[0] in self.all_answers:
                            idx = self.all_answers.index(answer[0])
                        else:
                            idx = -1
                        one_hot = [1.0 if j == idx else 0.0 for j in range(all_answers_len)]
                    elif self.mode == 'multi':
                        one_hot = [0.0 for j in range(all_answers_len)]
                        for x in answer[0].split(','):
                            if x in self.all_answers:
                                idx = self.all_answers.index(x)
                                one_hot[idx] = 1.0
                    graph.label.append(one_hot)
                #####

            # delete all params - keep architecture only
            for param in net.parameters():
                param.data = torch.tensor(param.shape).float()#.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            graph.model = net
            ######

            ### ni new items to be added to the graph
            if not graph.shape_ind:
                _, params_map = GHN._map_net_params_v2(graph)
                shape_ind = ShapeEncoder().eval().get_shape_info(graph.n_nodes, params_map)
                graph.shape_ind = shape_ind
            #print(shape_ind.shape)
            ###

            # zero-padd the graph elements - instead of the following, we will to batch-padding in SentenceTransformers
            #graph = graph_padding(graph, max_node_size=self.max_node_size, max_edge_size=self.max_edge_size)
            graph.edges = None

            #if(len(graph.texts)==1):
            if self.dataset_type == "arch2lang":
                # take pos-only samples
                neg_samples_index = graph.label.index(0.0)
            else:
                # take all pos and neg samples
                neg_samples_index = -1

            inp_example = InputExample(texts=graph.texts[:neg_samples_index],
                                       graph=[graph.node_feat, graph.shape_ind, graph.edges, graph._Adj], arch=graph,
                                       label=graph.label[:neg_samples_index], unique_layers=graph.unique_layers, n_params=graph.n_params, n_layers=graph.n_layers)

        return inp_example#, args['description']


    def _init_graph(self, A, nodes, layers):

        N = A.shape[0]
        assert N == len(nodes), (N, len(nodes))

        node_feat = torch.zeros(N, 1, dtype=torch.long)
        node_info = [[] for _ in range(layers)]
        param_shapes = []

        for node_ind, node in enumerate(nodes):
            name = self.primitives_ext[node[0]]
            cell_ind = node[1]
            name_op_net = self.op_names_net[node[2]]

            sz = None

            if not name_op_net.startswith('classifier'):
                # fix some inconsistency between names in different versions of our code
                if len(name_op_net) == 0:
                    name_op_net = 'input'
                elif name_op_net.endswith('to_out.0.'):
                    name_op_net += 'weight'
                else:
                    parts = name_op_net.split('.')
                    for i, s in enumerate(parts):
                        if s == '_ops' and parts[i + 2] != 'op':
                            try:
                                _ = int(parts[i + 2])
                                parts.insert(i + 2, 'op')
                                name_op_net = '.'.join(parts)
                                break
                            except:
                                continue

                name_op_net = 'cells.%d.%s' % (cell_ind, name_op_net)

                stem_p = name_op_net.find('stem')
                pos_enc_p = name_op_net.find('pos_enc')
                if stem_p >= 0:
                    name_op_net = name_op_net[stem_p:]
                elif pos_enc_p >= 0:
                    name_op_net = name_op_net[pos_enc_p:]
                elif name.find('pool') >= 0:
                    sz = (1, 1, 3, 3)  # assume all pooling layers are 3x3 in our DeepNets-1M

            if name.startswith('conv_'):
                if name == 'conv_1x1':
                    sz = (3, 16, 1, 1)          # just some random shape for visualization purposes
                name = 'conv'                   # remove kernel size info from the name
            elif name.find('conv_') > 0 or name.find('pool_') > 0:
                name = name[:len(name) - 4]     # remove kernel size info from the name
            elif name == 'fc-b':
                name = 'bias'

            param_shapes.append(sz)
            node_feat[node_ind] = self.primitives_dict[name]
            if name.find('conv') >= 0 or name.find('pool') >= 0 or name in ['bias', 'bn', 'ln', 'pos_enc']:
                node_info[cell_ind].append((node_ind, name_op_net, name, sz, node_ind == len(nodes) - 2, node_ind == len(nodes) - 1))

        A = torch.from_numpy(A).long()
        A[A > self.virtual_edges] = 0
        assert A[np.diag_indices_from(A)].sum() == 0, (
            'no loops should be in the graph', A[np.diag_indices_from(A)].sum())

        graph = Graph(node_feat=node_feat, node_info=node_info, A=A)
        graph._param_shapes = param_shapes

        return graph

    def _get_predefined(self):

        graphs = []
        for idx, arch in enumerate(['resnet50', 'ViT']):  # Here, other torchvision models can be added

            num_classes = 1000 if self.large_images else 10  # the exact number should not be important at the graph construction stage
            if arch == 'resnet50':
                model = adjust_net(eval('torchvision.models.%s(num_classes=%d)' % (arch, num_classes)),
                                   large_input=self.large_images)
                args = {'genotype': arch}
            else:
                args = {'C': 128,
                        'genotype': eval('genotypes.%s' % arch),
                        'n_cells': 12,
                        'glob_avg': True,
                        'preproc': False,
                        'C_mult': 1}
                model = Network(num_classes=num_classes, is_imagenet_input=self.large_images, **args)

            graphs.append(Graph(model, net_args=args, net_idx=idx))

        return graphs

class NetBatchSampler(torch.utils.data.BatchSampler):
    r"""
    Wrapper to sample batches of architectures.
    Allows for infinite sampling and filtering out batches not meeting certain conditions.
    """
    def __init__(self, deepnets, meta_batch_size=1):
        super(NetBatchSampler, self).__init__(
            torch.utils.data.RandomSampler(deepnets) if deepnets.is_train
            #torch.utils.data.SequentialSampler(deepnets) if deepnets.is_train
            else torch.utils.data.SequentialSampler(deepnets),
            meta_batch_size,
            drop_last=False)
        self.max_nodes_batch = MAX_NODES_BATCH if deepnets.is_train else None

    def check_batch(self, batch):
        return (self.max_nodes_batch is None or
                self.sampler.data_source.nodes[batch].sum() <=
                self.max_nodes_batch)

    def __iter__(self):
        while True:  # infinite sampler
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    if self.check_batch(batch):
                        yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                if self.check_batch(batch):
                    yield batch