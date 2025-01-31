# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils.

"""


import os
import random
import numpy as np
import time
import torch
import torchvision
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from torchvision.models import *
from ..deepnets1m.genotypes import DARTS
from .darts_utils import AvgrageMeter, accuracy, load_DARTS_pretrained


def pretrained_model(model, ckpt, num_classes, debug, ghn_class):
    """
    Loads the parameters from a trained checkpoint or predicts the parameters using GHNs.
    :param model: neural network (nn.Module)
    :param ckpt: path to the checkpoint of the model or GHN (ckpt=None for pretrained torchvision models)
    :param num_classes: number of classes in the target dataset (where the model will be trained)
    :param ghn_class: The GHN class
    :param debug: how much information to print
    :return: the model with pretrained/predicted parameters
    """
    if ckpt is not None:
        device = 'cpu'  # use CPU to load the model
        assert os.path.exists(ckpt), 'ckpt is specified ({}), but does not exist'.format(ckpt)
        state_dict = torch.load(ckpt, map_location=device)
        if 'config' in state_dict:
            ghn = ghn_class(**state_dict['config'], debug_level=debug).to(device).eval()
            ghn.load_state_dict(state_dict['state_dict'])
            # Predict all the parameters except for the last classification layers (in case the image dataset changes)
            ghn(model, predict_class_layers=ghn.num_classes == num_classes, bn_train=False)
        else:
            # Load a trained model
            if hasattr(model, 'genotype') and model.genotype == DARTS:
                model, res = load_DARTS_pretrained(model, checkpoint=ckpt, device=device,
                                                   load_class_layers=num_classes == 1000)
            else:
                state_dict = torch.load(ckpt, map_location=device)['state_dict']
                params = dict(model.named_parameters())
                names = list(state_dict.keys())
                for name in names:
                    if name.startswith('classifier') and state_dict[name].shape != params[name].shape:
                        del state_dict[name]
                res = model.load_state_dict(state_dict, strict=False)
            print('loaded pretrained model from {} (result = {})'.format(ckpt, res))

    elif isinstance(model, torchvision.models.ResNet):
        if num_classes != 1000:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        print('loaded pretrained {} torchvision model'.format(model.__class__.__name__))
    else:
        raise NotImplementedError('model classification layer must be adjusted', type(model))

    return model


def adjust_net(net, large_input=False):
    """
    Adjusts the first layers of the network so that small images (32x32) can be processed.
    :param net: neural network
    :param large_input: True if the input images are large (224x224 or more).
    :return: the adjusted network
    """
    if not net.expected_image_sz:
        net.expected_image_sz = 224 if large_input else 32

    if large_input:
        return net

    def adjust_first_conv(conv1, ks=(3, 3), stride=1):
        assert conv1.in_channels == 3, conv1
        ks_org = conv1.weight.data.shape[2:]
        # ni
        #ks_org = torch.Size(conv1.kernel_size)
        if ks_org[0] > ks[0] or ks_org[1] or ks[1]:
            # use the center of the filters
            offset = ((ks_org[0] - ks[0]) // 2, (ks_org[1] - ks[1]) // 2)
            offset1 = ((ks_org[0] - ks[0]) % 2, (ks_org[1] - ks[1]) % 2)
            conv1.weight.data = conv1.weight.data[:, :, offset[0]:-offset[0]-offset1[0], offset[1]:-offset[1]-offset1[1]]
            assert conv1.weight.data.shape[2:] == ks, (conv1.weight.data.shape, ks)
        conv1.kernel_size = ks
        conv1.padding = (ks[0] // 2, ks[1] // 2)
        conv1.stride = (stride, stride)

    if isinstance(net, ResNet):

        adjust_first_conv(net.conv1)
        assert hasattr(net, 'maxpool'), type(net)
        net.maxpool = nn.Identity()

    elif isinstance(net, DenseNet):

        adjust_first_conv(net.features[0])
        assert isinstance(net.features[3], nn.MaxPool2d), (net.features[3], type(net))
        net.features[3] = nn.Identity()

    elif isinstance(net, (MobileNetV2, MobileNetV3)):  # requires torchvision 0.9+

        def reduce_stride(m):
            if isinstance(m, nn.Conv2d):
                m.stride = 1

        for m in net.features[:5]:
            m.apply(reduce_stride)

    elif isinstance(net, VGG):

        for layer, mod in enumerate(net.features[:10]):
            if isinstance(mod, nn.MaxPool2d):
                net.features[layer] = nn.Identity()

    elif isinstance(net, AlexNet):

        net.features[0].stride = 1
        net.features[2] = nn.Identity()

    elif isinstance(net, MNASNet):

        net.layers[0].stride = 1

    elif isinstance(net, ShuffleNetV2):

        net.conv1.stride = 1
        net.maxpool = nn.Identity()

    elif isinstance(net, GoogLeNet):

        net.conv1.stride = 1
        net.maxpool1 = nn.Identity()

    else:
        print('WARNING: the network (%s) is not adapted for small inputs which may result in lower performance' % str(
            type(net)))

    return net


def set_seed(seed, only_torch=False):
    if not only_torch:
        random.seed(seed)  # for some libraries
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capacity(model):
    c, n = 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            c += 1
            n += np.prod(p.shape)
    return c, int(n)


def rand_choice(x, n=None):
    return x[torch.randint(len(x) if n is None else min(n, len(x)), size=(1,))]


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
