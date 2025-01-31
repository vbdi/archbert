import gc
import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
import torch

from transformers import AutoModel, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelWithLMHead
import numpy as np
import random
#from memory_profiler import profile as pppp

import functools
from streamlit import caching

def set_seed(seed, only_torch=False):
    if not only_torch:
        random.seed(seed)  # for some libraries
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import pickle
import torch
import pandas as pd
import torchvision
from graph.load_graph import load
import pprint
#from flops_calc.thop import profile
import sys

all_main_dic = {
    'AdaptiveAvgPool2d': '2D Adaptive Average pooling layer',# that calculates the correct kernel size for producing an output of the given dimensionality from the given input',
    'AnchorGenerator': 'Anchor Generator module which is a standard for 2D anchor-based detectors',
    'AvgPool2d': '2D average pooling layer used for calculating the average for each patch of the feature map',
    'BatchNorm2d': '2D Batch Normalization layer applied over a 4D input',# (a mini-batch of 2D inputs with additional channel dimension)',
    # 'Conv2d': '2D convolutional layer, which creates a convolution kernel convolved with the layer input',# to produce a tensor of outputs',
    'ConvTranspose2d': '2D transposed convolution layer that applies convolution with a fractional stride',
    'DeQuantStub': 'dequantization module which converts tensors from quantized to floating point',
    'DefaultBoxGenerator': 'A default box generator module used for SSD that operates similarly to the FasterRCNN AnchorGenerator',
    'Dropout': 'Dropout layer that is used to drastically reduce the chance of overfitting during training',
    'FrozenBatchNorm2d': '2D frozen batch normalization module in which the batch statistics and the affine parameters are fixed',
    'GELU': 'Gaussian Error Linear Units (gelu) activation function which is a smoother version of RELU',# in which the nonlinearity weights inputs by their percentile',
    'GeneralizedRCNNTransform': 'generalized rcnn transform module which performs input transformation before feeding the data to a GeneralizedRCNN model',
    'Hardsigmoid': 'hardsigmoid activation function that is a piece-wise linear approximation of the sigmoid function',
    'Hardswish': 'Hard Swish activation function that replaces the computationally expensive sigmoid with a piecewise linear analogue',
    'LastLevelMaxPool': 'last level max pooling layer applied on top of the last feature map',
    'LayerNorm': 'Layer normalization over input across the features instead of batch dimension',
    'LayerNorm2d': 'Layer normalization over input across the features instead of batch dimension',
    'Linear': 'linear module which applies a linear transformation to the incoming data',
    'MaxPool2d': '2d max pooling layer which is a pooling operation that calculates the maximum value',#, or largest, value in each patch of each feature map',
    'MultiScaleRoIAlign': 'Multi-scale RoIAlign pooling module, which is useful for detection with or without FPN',
    'NonDynamicallyQuantizableLinear': 'non dynamically quantizable linear module used to avoid triggering an obscure error when scripting an improperly quantized attention layer',
    'QuantStub': 'Quantize stub module that is a place holder for quantize operation',# that is used whenever the activation in the code crosses the quantized and non-quantized boundary',
    'ReLU': 'Rectified Linear Unit (relu) activation function which is a piecewise function',# that will output the input directly if it is positive, otherwise, it will output zero',
    'ReLU6': 'ReLU6 activation function which is a modification of the rectified linear unit (relu) where the activation is limited to a maximum size of 6',
    'SiLU': 'Sigmoid Linear Units (silu) activation function that is computed by the sigmoid function multiplied by its input',
    'Sigmoid': 'sigmoid activation function, also called the logistic function that is used to transform the input into a value between 0.0 and 1.0',
    'StochasticDepth': 'stochastic depth layer which aims to shrink the depth of a network during training',# while keeping it unchanged during testing',
    'UpsampleBilinear2D': 'UpsampleBilinear2D module that uses distance-weighted average of the four nearest pixel values to estimate a new pixel value',
    'PosEnc': 'A PosEnc module which does Relative Positional Encoding for Linear Attention Models',
    'Conv2d': 'convolutional layer, which creates a convolution kernel convolved with the layer input',
    'Sep_conv2d': 'Separable Convolution which divides a single convolution into two or more convolutions to reduce the number of parameters while producing the same output',
    'Dil_conv2d': 'Dilated convolution with a wider kernel created by inserting spaces between the kernel elements'

}

stop_models = ['Flatten', 'Permute', 'Identity', 'Zero', 'PosEnc']


def get_children(module, names_only, with_paths, layer_count, a_l_name_only):
    """
    Receives a module and returns a list of all layers including nested last level layers.
    :param module:
    :return:
    """
    # get children form module!
    children = list(module.children())

    if children == []:
        if str(module.__class__.__name__) != 'Sequential':

            if str(module.__class__.__name__) == 'Conv2d':

                if module.groups > 1:
                    if min(module.dilation) > 1:
                        a_l_name_only['Dil_conv2d'] = module.kernel_size
                        for_layer_count = 'Dil_conv2d'
                    else:
                        a_l_name_only['Sep_conv'] = module.kernel_size
                        for_layer_count = 'Sep_conv2d'
                else:
                    a_l_name_only['Conv2d'] = module.kernel_size
                    for_layer_count = 'Conv2d'

                if for_layer_count not in layer_count:
                    layer_count[for_layer_count] = [1]
                    layer_count[for_layer_count].append(module)
                else:
                    layer_count[for_layer_count][0] += 1
                    layer_count[for_layer_count].append(module)

            else:

                if module.__class__.__name__ not in layer_count:
                    layer_count[module.__class__.__name__] = [1]
                    layer_count[module.__class__.__name__].append(module)
                else:
                    layer_count[module.__class__.__name__][0] += 1
                    layer_count[module.__class__.__name__].append(module)

                if module.__class__.__name__ not in names_only:

                    # to avoid duplicated modules
                    if str(module.__class__.__name__).lower() in names_only.values():
                        names_only[module.__class__.__name__] = str(module.__class__.__name__).lower() + '_x'
                    else:
                        names_only[module.__class__.__name__] = str(module.__class__.__name__).lower()

                    if str(module.__class__.__name__).lower() in with_paths.values():
                        with_paths[module.__class__] = str(module.__class__.__name__).lower() + '_x'
                    else:
                        with_paths[module.__class__] = str(module.__class__.__name__).lower()

                    if str(module.__class__.__name__) != 'Conv2d':
                        a_l_name_only[module.__class__.__name__] = str(module.__class__.__name__).lower()

        # if module has no children; module is last child! :O
        return [module]

    else:
        # look for children from children... to the last child!
        for child in children:

            if str(child.__class__.__name__) == 'Conv2d':

                if child.groups > 1:
                    if min(child.dilation) > 1:
                        a_l_name_only['Dil_conv2d'] = child.kernel_size
                        for_layer_count = 'Dil_conv2d'
                    else:
                        a_l_name_only['Sep_conv2d'] = child.kernel_size
                        for_layer_count = 'Sep_conv2d'
                else:
                    a_l_name_only['Conv2d'] = child.kernel_size
                    for_layer_count = 'Conv2d'

                if for_layer_count not in layer_count:
                    layer_count[for_layer_count] = [1]
                    layer_count[for_layer_count].append(child)
                else:
                    layer_count[for_layer_count][0] += 1
                    layer_count[for_layer_count].append(child)

                if child.__class__.__name__ not in names_only:

                    if str(child.__class__.__name__).lower() in names_only.values():
                        names_only[child.__class__.__name__] = str(child.__class__.__name__).lower() + '_x'
                    else:
                        names_only[child.__class__.__name__] = str(child.__class__.__name__).lower()

                    if str(child.__class__.__name__).lower() in with_paths.values():
                        with_paths[child.__class__] = str(child.__class__.__name__).lower() + '_x'
                    else:
                        with_paths[child.__class__] = str(child.__class__.__name__).lower()

            else:

                if child.__class__.__name__ in layer_count:
                    layer_count[child.__class__.__name__][0] += 1
                    layer_count[child.__class__.__name__].append(child)

                if child.__class__.__name__ not in names_only:
                    get_children(child, names_only, with_paths, layer_count, a_l_name_only)


# @functools.lru_cache(maxsize = None)
# @pppp
def get_single_model_layers(model_name, names_only, with_paths, count, a_l_name_only, hugging_face = False):
    """
    receives a model name from input and returns a tuple containing two dictionaries. One with the layers
    path and one with the layers name only.
    :param model_name:
    :return:
    """

    print(model_name)
    if hugging_face:
        try:
            model = AutoModel.from_pretrained(model_name)
        except:
            print(f'could not load the  {model_name}')
            with open('problematic_HF_models.txt', 'a') as f:
                f.write(model_name + '\n')
            return
    else:
        model = (eval('torchvision.models.%s()' % model_name)).eval()

    model_layers = []

    for _, layer in model.named_modules():
      model_layers.append(layer)
      break

    del model
    model = None

    torch._C._cuda_emptyCache()
    get_children(model_layers[0], names_only, with_paths,count,a_l_name_only)
    del model_layers
    gc.collect()
    return



def get_all_model_layers(dataFrame, hugging_face = False):
    """
    receives a dataframe name from input and returns a list of tuple containing two dictionaries. One with the layers
    path and one with the layers name only.
    :param dataFrame:
    :return:
    """

    if hugging_face == True:

        layer_path = 'layers.pth'
    else:
        layer_path = 'layers.pth'


    all_model_names = load(dataFrame=dataFrame, load_from_path=False, unique_only=True)

    dict_model_names, with_paths, names_only = load_layers_dict(hugging_face= hugging_face)

    # Getting new models
    a = set(all_model_names)
    # b = set(dict_model_names)
    b = set()
    for item in dict_model_names:
        b.add(item)

    model_names = list(a.symmetric_difference(b))
    count = {}
    a_l_name_only = {}
    processed_models = [item for item in dict_model_names]

    # counter = 1
    for i in range(len(model_names)): #200:300  - 300:400 - 400:500 - 500:600 - 600:700 - 700:
        model_name = model_names[i]
        processed_models.append(model_name)
    # for model_name in model_names:
        if model_name not in ['quantization.shufflenet_v2_x1_5', 'quantization.shufflenet_v2_x2_0']:
            print(f'{i} : ', end='')
            gc.collect()
            get_single_model_layers(model_name, names_only, with_paths, count, a_l_name_only , hugging_face = hugging_face)
            gc.collect()
            # counter += 1

    stored_layers = {'model_names': processed_models, 'with_paths': with_paths, 'names_only': names_only}



    with open(layer_path, 'wb') as f:
        pickle.dump(stored_layers, f)

    return with_paths, names_only


def load_layers_dict(hugging_face= False):

    if hugging_face:
        dict_file_path = 'layers.pth'
    else:
        dict_file_path = 'layers.pth'

    if os.path.exists(dict_file_path):
        stored_layers = pickle.load(open(dict_file_path, 'rb'))
        with_paths = stored_layers['with_paths']
        names_only = stored_layers['names_only']
        model_names = stored_layers['model_names']
    else:
        model_names = []
        with_paths = {}
        names_only = {}

    return model_names, with_paths, names_only


######################################################################################################


def extract_layers_for_a_single_model(model):
    """
    Receives a Pytorch model as input and returns two dictionaries containing the path and names of each layer.
    :param Pytorch model from torchvision:
    :return: A tuple containing two dictionaries: one with the layer's path and one with layer's name only.
    """

    path = {}
    name = {}
    layer_count = {}
    a_l_name_only = {}
    model_layers = []

    for _, layer in model.named_modules():
      model_layers.append(layer)

    get_children(model_layers[0], path, name,layer_count,a_l_name_only)

    return path, name, layer_count, a_l_name_only


####################################################################################################

def auto_label_generator(model, num_params = 0, main_dic=all_main_dic):
    all_descriptions = []
    all_scores = []
    layers_with_name, layers_with_path, layer_count, a_l_name_only = extract_layers_for_a_single_model(model)
    list_of_positive_layer = list(a_l_name_only.keys())

    ############################## Added for testing############################
    for item in list_of_positive_layer:
        if item not in main_dic:

            # stopped models is defined on the top of this file
            if item not in stop_models:
                with open('problematic layers.txt', 'a') as f:
                    f.write(item + '\n')
                stop_models.append(item)
    ############################## Added for testing############################

    list_of_main_dic_layers = list(main_dic.keys())
    list_of_negative_layers = list(set(list_of_main_dic_layers).symmetric_difference(set(list_of_positive_layer)))
    list_of_positive_layer.append('Parameters')
    list_of_positive_layer.append('LayerCount')
    list_of_positive_layer.append('Total')
    # params = calculate_number_of_params(num_params)
    params = num_params
    pos_descriptions = generate_descriptions(list_of_positive_layer, layer_count, params, main_dic)

    for des in pos_descriptions:
        all_descriptions.append(des)
        all_scores.append(1.0)

    neg_descriptions = generate_descriptions(list_of_negative_layers, layer_count, params, main_dic)

    for des in neg_descriptions:
        all_descriptions.append(des)
        all_scores.append(0.0)

    # return descriptions, labels (scores)
    return all_descriptions, all_scores


def calculate_number_of_params(model):
    input = torch.randn(1, 3, 32, 32)
    _, num_of_params = profile(model, inputs=(input,))

    params = round(num_of_params / 1000000)

    res = ''
    if params > 1000:
        res = str(round(params / 1000)) + ' billion'
    elif params > 1:
        res = str(params) + ' million'
    elif params < 1:
        res = str(round(num_of_params / 10000)) + '000'

    return res

def make_overall_layer_count(layer_count,count):
    # total_count = sum(list(layer_count.values()))
    total_count = 0
    for item in layer_count:
        total_count += layer_count[item][0]

    if count %2 == 0:
        temp_text = f'In Totall, this neural network architecture has {total_count} layers '
    else:
        temp_text = f' and, in totall, it has {total_count} layers '

    if count == 2:
        temp_text += '.'
    return temp_text

def make_layer_count_descriptions(layer_count,count):
    keys = list(layer_count.keys())
    temp_text = ''
    for i, model in enumerate(keys):
        temp_text += str(layer_count[model][0])
        temp_text += ' '
        temp_text += model
        if i <= len(keys) - 3:
            temp_text += ', '
        elif i == len(keys) - 2:
            temp_text += ', and '

    temp_text += ' layers'

    if count % 2 == 0:
        temp_text2 = 'This model has ' + temp_text

    else:
        temp_text2 = ' and, it has ' + temp_text

    if count == 2:
        temp_text2 += '.'
    return temp_text2

def make_parameters_description(params, count):
    description = ''

    temp_text = f'about {round(params / 1000000,2)}'

    if  count % 2 == 0 or count == 2:
        description += f'This neural architecture has {temp_text} Million parameters'
    else:
        description += f' and has {temp_text} Million parameters.'

    return description

def generate_descriptions(list_of_layer_s_name,layer_count, params = 0, main_dic=all_main_dic):
    fill = [' It also has ', ' In addition, it has ', ' Additionally, this architecture contains ',
            ' The model also includes ', ' It also contains ',
            ' Another part of this neural network is ',
            ' This classifier also includes ']

    starter = ['model', 'neural network', 'network', 'neural architecture', 'architecture',
               'neural network model', 'neural network architecture',
               'classifier', 'classification model', 'classification neural network',
               'classification network', 'classification neural architecture', 'classification architecture',
               'classification neural network model', 'classification neural network architecture',
               'classification structure']
    random.shuffle(starter)

    starter_dic = {'1This ': ' has ', '1In this ': ' there exists ', '2In this ': ' there is ', '2This ': ' includes ',
                   '3This ': ' contains '}
    starter_list = list(starter_dic.keys())

    temp = []
    temp_var_for_starter_1 = []
    temp_var_for_starter_2 = []

    random.shuffle(list_of_layer_s_name)
    count = 0
    description = ''
    generated_descriptions = []
    for i, name in enumerate(list_of_layer_s_name):
        if name in stop_models: continue
        if name == 'LayerCount':
            description += make_layer_count_descriptions(layer_count,count)
            count += 1
        elif name == 'Total':
            description += make_overall_layer_count(layer_count,count)
            count += 1
        elif name == 'Parameters':

            description += make_parameters_description(params,count)

            count += 1
        else:

            des = main_dic[name]

            # if des[0].lower() in ['aeoiu']:
            #     a = False
            # else:
            #     a = True

            if count < 3:

                if count == 0:
                    if len(temp_var_for_starter_1) == len(starter_list):
                        temp_var_for_starter_1 = []
                    else:
                        holder1 = random.choice(starter_list)
                        while holder1 in temp_var_for_starter_1:
                            holder1 = random.choice(starter_list)
                        temp_var_for_starter_1.append(holder1)

                    description += holder1[1:]
                    if len(temp_var_for_starter_2) == len(starter):
                        temp_var_for_starter_2 = []
                    else:
                        holder2 = random.choice(starter)
                        while holder2 in temp_var_for_starter_2:
                            holder2 = random.choice(starter)
                        temp_var_for_starter_2.append(holder2)
                    description += holder2
                    description += starter_dic[holder1]

                else:
                    if count % 2 != 0:
                        description += ', and '
                    else:

                        if len(temp) == len(fill):
                            temp = []
                        else:
                            x = random.choice(fill)
                            while x in temp:
                                x = random.choice(fill)
                            temp.append(x)
                        description += x

                # if a:
                #     description += 'a '
                # else:
                #     description += 'an '

                description += des
                if count % 2 != 0 or count == 2 or i == len(list_of_layer_s_name) - 1:
                    description += '.'

                count += 1

            if count >= 3 or i == len(list_of_layer_s_name) - 1:
                generated_descriptions.append(description)
                description = ''
                temp = []
                count = 0

    if description != '' and description not in generated_descriptions:
        generated_descriptions.append(description)

    return generated_descriptions

# def complete_df_descriptions(dataFrame):
#     for idx,t in enumerate(dataFrame.text):
#         if t is None:

def n_limited_question_generator(layer_count,number_of_samples):
    #set_seed(100)

    what = ['what', 'which']
    type_of = ['type of', 'kind of']
    layer_module = ['layer','module','']
    layer_type = ['pooling', 'normalization', 'convoloution', 'activation']
    is_used = ['is used', 'exist', 'there exist', 'is included', 'has been used']
    model = ['model', 'neural network', 'network', 'neural architecture', 'architecture', 'neural network model',
             'neural network architecture']
    is_not_used = ['is not used', 'does not exist', 'does not include', 'has not been used']

    does = ['does', 'calculates', 'performs']

    modules_with_kernel_size = ['MaxPool2d', 'AvgPool2d', 'Conv2d', 'Sep_conv2d', 'Dil_conv2d', 'overall']

    names = {'MaxPool2d': ['2d max pooling', 'MaxPool2d', '2d max pool'],
             'AvgPool2d': ['2d average pooling', 'AvgPool2d', '2d average pool'],
             'AdaptiveAvgPool2d': ['2d adaptive average pooling', 'AdaptiveAvgPool2d', '2d adaptive average pool'],
             'LastLevelMaxPool' : ['LastLevelMaxPool','last level max pooling'],

             'BatchNorm2d' : ['BatchNorm2d','2d batch normalization','2d batch norm', '2d BatchNorm'],
             'LayerNorm': ['LayerNorm','layer normalization'],
             'FrozenBatchNorm2d':['FrozenBatchNorm2d','frozen batch normalization','2d frozen batch normalization'],
             'LayerNorm2d': ['LayerNorm2d','2d layer normalization'],

             'Conv2d': ['Conv2d','2d Convolution'],
             'Sep_conv2d':['Sep_conv2d','Separable Convolution','2d Separable Convolution'],
             'Dil_conv2d':['Dil_conv2d','Dilated Convolution','2d Dilated Convolution'],

             'GELU':['GELU','Gaussian Error Linear Unit'],
             'ReLU':['ReLU','Rectified Linear Unit'],
             'Hardswish':['Hardswish', 'hard swish'],
             'ReLU6':['ReLU6','Gaussian Error Linear Unit 6', 'Gaussian Error Linear Unit version 6'],
             'SiLU':['SiLU','sigmoid weighted linear unit'],
             'Sigmoid':['Sigmoid'],
             'Hardsigmoid':['Hardsigmoid','hard sigmoid']
             }

    pooling = {'MaxPool2d': 'calculating the maximum value for each patch of the feature map',
               'AvgPool2d': 'calculating the average for each patch of the feature map',
               'AdaptiveAvgPool2d': 'calculating the correct kernel size to use to average each patch of the feature map',
               'LastLevelMaxPool': ''}

    normalization = {'BatchNorm2d': 'applying a 2D Batch Normalization layer over a 4D input',
                     'LayerNorm': 'Layer normalization over input across the features instead of batch dimension',
                     'FrozenBatchNorm2d': '',
                     'LayerNorm2d': ''}

    convoloution = {'Conv2d': 'creating a convolution kernel and convolve it with the layer input',
                    'Sep_conv2d': 'dividing single convolutions into two or more and reducing number of parameters while producing the same output',
                    'Dil_conv2d': 'creating a wider kernel by inserting spaces between the kernel elements'}

    activation = {'GELU': 'calculating Gaussian Error Linear Units (gelu) for a smoother version of RELU',
                  'ReLU': 'calculating Rectified Linear Unit (relu) which is a piecewise function',
                  'Hardswish': 'replacing the computationally expensive sigmoid with a piecewise linear analogue',
                  'ReLU6': '',
                  'SiLU': '',
                  'Sigmoid': '',
                  'Hardsigmoid': ''}

    followed = ['is followed by', 'is used after', 'comes along with', 'comes after']

    overall = ['overall', 'in total', 'in general']
    are_used = ['are used', 'exist', 'there exist', 'are included', 'have been used']

    kernel_size = extract_kernel_size(layer_count)

    all_questions = []
    all_answers = []

    for _ in range(number_of_samples):

        # Type 1

        for layer in layer_type:

            question = f'{random.choice(what)} {random.choice(type_of)} {layer} {random.choice(layer_module)} {random.choice(is_used)} in this {random.choice(model)}?'
            module = eval(layer)
            answer = [','.join(x for x in layer_count if x in module)]
            all_questions.append(question)
            all_answers.append(answer)

            # type 1- model 2
            if layer != 'convoloution':

                question = f'{random.choice(what)} {random.choice(type_of)} {layer} {random.choice(layer_module)} {random.choice(followed)} convolution ' \
                           f'{random.choice(layer_module)} in this {random.choice(model)}?'
                answer = [','.join([x for x in layer_count if x in module])]

                all_questions.append(question)
                all_answers.append(answer)

            # Type 2
            module = eval(layer)
            question = f'{random.choice(what)} {layer} {random.choice(layer_module)} {random.choice(is_not_used)} in this {random.choice(model)}?'
            answer = [','.join([x for x in module if x not in layer_count])]
            if answer[0] == '':
                answer = ['None']

            all_questions.append(question)
            all_answers.append(answer)

            # Type 3

            for item in module:
                question = f'what {random.choice(names[str(item)])} {random.choice(layer_module)} {random.choice(does)} in this {random.choice(model)}?'
                if item in layer_count:
                    answer = [module[item]]
                    if answer[0] == '':
                        answer = [f'This model does not include {item}']
                else:
                    answer = [f'This model does not include {item}']

                all_questions.append(question)
                all_answers.append(answer)

        # Type 4

        for m in modules_with_kernel_size:

            if m == 'overall':
                question = f'{random.choice(overall)} {random.choice(what)} kernel size {random.choice(are_used)} in this {random.choice(model)}?'
            else:
                question = f'{random.choice(what)} {random.choice(names[m])} kernel size {random.choice(is_used)} in this {random.choice(model)}?'

            if m == 'overall':
                answer = [','.join(list(kernel_size['overall']))]
            elif m in kernel_size:
                answer = [','.join([str(k) for k in kernel_size[m]])]

            else:
                answer = [f'This model does not include {m}']

            all_questions.append(question)
            all_answers.append(answer)

        # Type 5

        question = f'{random.choice(overall)} {random.choice(what)} {random.choice(type_of)} {random.choice(["layers","modules"])} {random.choice(are_used)} in this {random.choice(model)}?'
        answer = [','.join(list(layer_count.keys()))]

        all_questions.append(question)
        all_answers.append(answer)


    return all_questions, all_answers

def question_generator_2(layer_count):
    what = ['What', 'Which']
    type_of = ['type of', 'kind of']
    layer_type = ['pooling', 'normalization', 'convoloution', 'activation']
    is_used = ['is used', 'exist', 'there exist', 'is included', 'has been used']
    model = ['model', 'neural network', 'network', 'neural architecture', 'architecture', 'neural network model',
             'neural network architecture']
    is_not_used = ['is not used', 'does not exist', 'does not include', 'has not been used']
    layer_type_specific = None
    does = ['does', 'calculates', 'performes']

    modules_with_kernel_size = ['MaxPool2d', 'AvgPool2d', 'Conv2d', 'Sep_conv2d', 'Dil_conv2d', 'overall']

    pooling = {'MaxPool2d': 'calculating the maximum value for each patch of the feature map',
               'AvgPool2d': 'calculating the average for each patch of the feature map',
               'AdaptiveAvgPool2d': 'calculating the correct kernel size to use to average each patch of the feature map',
               'LastLevelMaxPool': ''}

    normalization = {'BatchNorm2d': 'applying a 2D Batch Normalization layer over a 4D input',
                     'LayerNorm': 'Layer normalization over input across the features instead of batch dimension',
                     'FrozenBatchNorm2d': '',
                     'LayerNorm2d': ''}

    convoloution = {'Conv2d': 'creating a convolution kernel and convolve it with the layer input',
                    'Sep_conv2d': 'dividing single convolutions into two or more and reducing number of parameters while producing the same output',
                    'Dil_conv2d': 'creating a wider kernel by inserting spaces between the kernel elements'}

    activation = {'GELU': 'calculating Gaussian Error Linear Units (gelu) for a smoother version of RELU',
                  'ReLU': 'calculating Rectified Linear Unit (relu) which is a piecewise function',
                  'Hardswish': 'replacing the computationally expensive sigmoid with a piecewise linear analogue',
                  'ReLU6': '',
                  'SiLU': '',
                  'Sigmoid': '',
                  'Hardsigmoid': ''}

    overall = ['overall', 'in total', 'in general']
    are_used = ['are used', 'exist', 'there exist', 'are included', 'have been used']
    kernel_size = extract_kernel_size(layer_count)

    all_questions = []
    all_answers = []

    # Type 1

    for i1 in layer_type:
        module = eval(i1)
        answer = [','.join([layer for layer in layer_count if layer in module])]
        for i2 in type_of:
            for i3 in what:
                for i5 in is_used:
                    for i6 in model:
                        question = f'{i3} {i2} {i1} layer {i5} in this {i6}?'
                        all_questions.append(question)
                        all_answers.append(answer)

    for i1 in layer_type:
        if i1 != 'convoloution':
            module = eval(i1)
            answer = [','.join([layer for layer in layer_count if layer in module])]
            for i2 in type_of:
                for i3 in what:
                    for i6 in model:
                        question = f'{i3} {i2} {i1} is followed by the convolution in this {i6}?'
                        all_questions.append(question)
                        all_answers.append(answer)

    # Type 2

    for i1 in layer_type:
        module = eval(i1)
        answer = [','.join([layer for layer in module if layer not in layer_count])]
        if answer[0] == '':
            answer = ['None']
        for i3 in what:
            for i5 in is_not_used:
                for i6 in model:
                    question = f'{i3} {i1} {i5} in this {i6}'
                    all_questions.append(question)
                    all_answers.append(answer)


    # Type 3

    for i1 in layer_type:
        for i0 in eval(i1):
            if i0 in layer_count:
                answer = [eval(i1)[i0]]
                if answer[0] == '':
                    answer = [f'This model does not include {i0}']
            else:
                answer = [f'This model does not include {i0}']

            for i6 in model:
                for i7 in does:
                    question = f'what {i0} layer {i7} in this {i6}?'
                    all_questions.append(question)
                    all_answers.append(answer)

    # Type 4

    for i1 in modules_with_kernel_size:
        if i1 == 'overall':
            answer = [','.join(list(kernel_size['overall']))]
        elif i1 in kernel_size:
            answer = [','.join([str(k) for k in kernel_size[i1]])]

        else:
            answer = [f'This model does not include {i1}']
        for i6 in model:
            for i3 in what:
                for i4 in are_used:
                    for i5 in overall:
                        if i1 == 'overall':
                            question = f'{i5} {i3} kernel size {i4} in this {i6}?'
                        else:
                            question = f'{i3} {i1} kernel size {i4} in this {i6}?'

                        all_questions.append(question)
                        all_answers.append(answer)

    # Type 5
    for i2 in type_of:
        answer = [','.join(list(layer_count.keys()))]
        for i3 in what:
            for i5 in are_used:
                for i6 in model:
                    question = f'{i3} {i2} layers {i5} in this {i6}?'
                    all_questions.append(question)
                    all_answers.append(answer)



    return all_questions, all_answers


def question_generator(layer_count):

    list_of_layers = list(layer_count.keys())

    kernel_size = extract_kernel_size(layer_count)

    all_questions = []
    all_answers = []
    pooling = {'MaxPool2d': 'calculating the maximum value for each patch of the feature map',
               'AvgPool2d': 'calculating the average for each patch of the feature map',
               'AdaptiveAvgPool2d': 'calculating the correct kernel size to use to average each patch of the feature map',
               'LastLevelMaxPool': ''}

    norm = {'BatchNorm2d': 'applying a 2D Batch Normalization layer over a 4D input',
            'LayerNorm': 'Layer normalization over input across the features instead of batch dimension',
            'FrozenBatchNorm2d': '',
            'LayerNorm2d': ''}

    conv = {'Conv2d': 'creating a convolution kernel and convolve it with the layer input',
            'Sep_conv2d': 'dividing single convolutions into two or more and reducing number of parameters while producing the same output',
            'Dil_conv2d': 'creating a wider kernel by inserting spaces between the kernel elements'}

    activation = {'GELU': 'calculating Gaussian Error Linear Units (gelu) for a smoother version of RELU',
                  'ReLU': 'calculating Rectified Linear Unit (relu) which is a piecewise function',
                  'Hardswish': 'replacing the computationally expensive sigmoid with a piecewise linear analogue',
                  'ReLU6': '',
                  'SiLU': '',
                  'Sigmoid': '',
                  'Hardsigmoid': ''}

    questions = {
        'What pooling layer is used in this model?': ('pooling', 1),
        'What type of normalization is used in this architecture?': ('norm', 1),
        'What type of activation is used in this neural network?': ('activation', 1),
        'What type of convolution is used in this neural network?': ('conv', 1),
        'What type of pooling layer is followed by the convolution in this neural network?': ('pooling', 1),
        'What type of normalization is followed by the convolution in this model?': ('norm', 1),
        'What type of activation is followed by the convolution in this architecture?': ('activation', 1),
        'What pooling layer is not used in this model': ('pooling', 2),
        'What normalization is not used in this architecture?': ('norm', 2),
        'What convolution is not used in this neural architecture?': ('conv', 2),
        'What activation is not used in this neural network model?': ('activation', 2),
        'What does the max pooling layer do in this model?': ('pooling', 3, 'MaxPool2d'),
        'What does the average pooling layer do in this model?': ('pooling', 3, 'AvgPool2d'),
        'What does the adaptive average pooling layer do in this model?': ('pooling', 3, 'AdaptiveAvgPool2d'),
        'What does the 2D batch normalization layer do in this architecture?': ('norm', 3, 'BatchNorm2d'),
        'What does the 2D layer normalization do in this architecture?': ('norm', 3, 'LayerNorm2d'),
        'What does the 2D convolution layer do in this neural network?': ('conv', 3, 'Conv2d'),
        'What does the Separable convolution layer do in this neural network?': ('conv', 3, 'Sep_conv2d'),
        'What does the Dilated convolution layer do in this neural network?': ('conv', 3, 'Dil_conv2d'),
        'What does the GELU activation  layer do in this neural network?': ('activation', 3, 'GELU'),
        'What does the ReLU activation  layer do in this neural network?': ('activation', 3, 'ReLU'),
        'What does the Hardswish activation  layer do in this neural network?': ('activation', 3, 'Hardswish'),
        'What convolution kernel size is used in this model?': ('conv', 4, 'Conv2d'),
        'What separable convolution kernel size is used in this model?': ('conv', 4, 'Sep_conv2d'),
        'What dilated convolution kernel size is used in this model?': ('conv', 4, 'Dil_conv2d'),
        'What max pooling kernel size is used in this model?': ('pooling', 4, 'MaxPool2d'),
        'What average pooling kernel size is used in this model?': ('pooling', 4, 'AvgPool2d'),
        'What kernel sizes are used in this model?': ('pooling', 4, 'overall'),
        'What kind of layers used in this architecture?': (None, 5, None)
    }

    for q in questions:

        question = q
        if questions[q][1] == 1:
            module = eval(questions[q][0])
            answer = [','.join([layer for layer in list_of_layers if layer in module])]
            if len(answer[0]) == 0:
                answer = ['None']

        elif questions[q][1] == 2:
            module = eval(questions[q][0])
            answer = [','.join([layer for layer in module if layer not in list_of_layers])]
            if len(answer[0]) == 0:
                answer = ['None']
            # all_questions.append((question, answer))


        elif questions[q][1] == 3:
            answer = []
            module = eval(questions[q][0])
            module_type = questions[q][2]

            if module_type in list_of_layers:
                s = module[module_type] + f' by {module_type}'
                answer.append(s)
            else:
                s = f"This model does not include {module_type}"
                answer.append(s)

            # all_questions.append((question, answer))

        elif questions[q][1] == 4:
            module = questions[q][2]
            if module in kernel_size:
                answer = [','.join([str(k) for k in kernel_size[module]])]
            else:
                answer = [f"This model does not include {module}"]
            # all_questions.append((question, answer))


        elif questions[q][1] == 5:
            answer = [','.join(list(layer_count.keys()))]
            # all_questions.append((question, answer))

        all_questions.append(question)
        all_answers.append(answer)

    return all_questions, all_answers


def extract_kernel_size(layer_count):
    d = {}

    for item in layer_count:
        x = layer_count[item]

        for i in range(1,len(x)):
            if hasattr(x[i], 'kernel_size'):
                if item not in d:
                    d[item] = set()

                k = x[i].kernel_size
                if isinstance(k, int):

                    kernel = str(k) + '*' + str(k)
                else:
                    kernel = str(k[0]) + '*' + str(k[0])

                d[item].add(kernel)
        overall = set()
        for item in d:
            for k in d[item]:
                overall.add(k)
        d['overall'] = overall

    return d

def restore_layer_count_only(layer_count):
    count = {}
    for item in layer_count:
        count[item] = layer_count[item][0]

    return count
