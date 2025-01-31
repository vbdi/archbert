import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch

import pandas as pd
import sys
import torch
import pickle
import glob
import random
from transformers import AutoTokenizer
from os.path import exists

sys.path.append('../..')

# def load_clone_ds(bimodal=False, unique_only=True):
#     if unique_only:
#         print("test")


def load_clone_ds(dataFrame,load_path = '', bimodal=False, num_nets=None, shuffle=False):

    stop_models = ['video.r3d_18', 'video.mc3_18', 'video.r2plus1d_18']
    graph_files = glob.glob(load_path)
    graphs_list = []
    data = pd.read_csv(dataFrame, encoding="UTF-8")
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)

    # unique_models = data.arch1.unique()


    graph_dictionary = {}

    with torch.no_grad():
        for idx, file_path in enumerate(graph_files):
            #if num_nets is not None and idx >= num_nets:
            #    break
            file_name = file_path.split('/')[-1]
            file_name = file_name[:-6]
            pure_model_name = file_name.split('___')[-1]
            if exists(file_path):
                graph_file = open(file_path, 'rb')
                g = pickle.load(graph_file)
                graph_dictionary[pure_model_name] = (file_name,file_path,g)

    # ni
    if num_nets is None or num_nets >= len(data):
        num_nets = len(data)
    if shuffle:
        data = data.sample(frac=1, random_state=100).reset_index()

    for row_idx in range(num_nets):
        arch_1 = data.loc[row_idx,'arch1']
        arch_2 = data.loc[row_idx,'arch2']
        h_score = data.loc[row_idx, 'hard_score']

        arch_1 = arch_1.split('/')[-1]
        arch_2 = arch_2.split('/')[-1]

        if arch_1 in graph_dictionary and arch_2 in graph_dictionary:
            arch_name_1 = graph_dictionary[arch_1][0]
            arch_name_2 = graph_dictionary[arch_2][0]

            # graph_file_1 = open(graph_dictionary[arch_1][1], 'rb')
            # graph_file_2 = open(graph_dictionary[arch_2][1], 'rb')

            g_1 = graph_dictionary[arch_1][2]
            g_2 = graph_dictionary[arch_2][2]

            if bimodal:
                arch1_txt = data.loc[row_idx, 'm1_text']
                arch2_txt = data.loc[row_idx, 'm2_text']

                s_score = data.loc[row_idx, 'soft_score']

                graphs_list.append([g_1,g_2,h_score,arch_name_1,arch_name_2,arch1_txt,arch2_txt,s_score])

            else:
                graphs_list.append([g_1, g_2, h_score, arch_name_1, arch_name_2])

    return graphs_list



def load(dataFrame='',
         load_path='', load_from_path=True, unique_only=False, return_vocab=False, tokenizer_model_path="", num_nets=None):
    # arch = torch.jit.load('arch_no_param_'+str(idx)+'.pth')

    stop_models = ['video.r3d_18', 'video.mc3_18', 'video.r2plus1d_18']

    graph_files = glob.glob(load_path)

    graphs_list = []

    data = pd.read_csv(dataFrame, encoding="UTF-8")

    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis=1, inplace=True)

    if unique_only:
        unique_data = pd.DataFrame(columns=data.columns)
        index = 0
        for d_name in data.name.unique():
            if d_name not in stop_models:
                unique_data.loc[index] = data[data.name == d_name].iloc[0]
                index += 1
    ########

    the_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_path)
    list_of_words = []
    for i in range(len(data)):
        text = data.loc[i,'text']
        temp = the_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        for item in temp:
            list_of_words.append(item[0].lower())
        # print(list_of_words)

    list_of_words = set(list_of_words)
    new_vocab_list = list(list_of_words)

    ########
    # if return_vocab:
    #     new_vocab_list = ["ali","gholi3"]

    if load_from_path:

        with torch.no_grad():
            for idx, file_path in enumerate(graph_files):
                if num_nets is not None and idx >= num_nets:
                    break
                '''
                pytorch_model = torch.load(file_path + "_" + str(idx) + ".pth")
                # arch = torch.jit.load('arch_no_param_'+str(idx)+'.pth')
                for param in pytorch_model.parameters():
                    # initilize all params with zeros (based on their original shape)
                    param.data = torch.zeros(torch.Size(param.data.clone().detach().int()))  # param.data #torch.tensor(0)
                pytorch_models.append(pytorch_model)
                '''

                # Extract the file name without .graph from the path
                file_name = file_path.split('/')[-1]
                file_name = file_name[:-6]

                ### load graphs
                # if load_graphs:
                graph_file = open(file_path, 'rb')
                g = pickle.load(graph_file)
                # data2 = data[data.folder_name == file_name].copy()
                # data2.reset_index(inplace=True)
                if unique_only:
                    #### modified by ni - to be used for evaluation ####
                    temp_data = data[data.folder_name == file_name]
                    temp_data.reset_index(inplace=True)
                    texts=[]
                    socres=[]
                    for jdx, text in enumerate(temp_data.text):
                        texts.append(text)
                        socres.append(temp_data.loc[jdx,'score'])
                    ##############
                    graphs_list.append([g, texts, socres, file_name])
                    # graphs_list.append([g, data2.loc[0, 'text'], data2.loc[0, 'score'], data2.loc[0, 'name']])

                    # ni for clone-detection no-text dataset
                    # graphs_list.append([g1, g2, hard_socre, file_name1, file_name2])

                    # ni for clone-detection with-text dataset
                    # graphs_list.append([g1, g2, hard_socre, file_name1, file_name2, texts_part1, texts_part2, soft_socres])

                else:
                    data2 = data[data.folder_name == file_name].copy()
                    data2.reset_index(inplace=True)
                    for jdx, text in enumerate(data2.text):
                        graphs_list.append([g, text, data2.loc[jdx,'score'], data2.loc[jdx,'name']])

                        # ni for clone-detection no-text dataset
                        #graphs_list.append([g1, g2, score, data2.loc[jdx, 'name1'], data2.loc[jdx, 'name2']])

                        # ni for clone-detection with-text dataset
                        #graphs_list.append([g1, g2, hard_score, data2.loc[jdx, 'name1'], data2.loc[jdx, 'name2'], texts_part1, texts_part2, data2.loc[jdx, 'soft_scores']])
    else:
        if unique_only:
            unique_names = unique_data.name
            for name in unique_names:
                graphs_list.append(name)

    if return_vocab:
        return graphs_list, new_vocab_list
    else:
        return graphs_list

def load_old(dataFrame, load_path, load_graphs=True, unique_graphs=True):
    # arch = torch.jit.load('arch_no_param_'+str(idx)+'.pth')
    data = pd.read_csv(dataFrame, encoding="UTF-8")

    graph_files = glob.glob(load_path)

    pytorch_models = []
    graphs_list = []
    with torch.no_grad():
        for idx, file_path in enumerate(graph_files):
            '''
            pytorch_model = torch.load(file_path + "_" + str(idx) + ".pth")
            # arch = torch.jit.load('arch_no_param_'+str(idx)+'.pth')
            for param in pytorch_model.parameters():
                # initilize all params with zeros (based on their original shape)
                param.data = torch.zeros(torch.Size(param.data.clone().detach().int()))  # param.data #torch.tensor(0)
            pytorch_models.append(pytorch_model)
            '''
            file_name = file_path.split('/')[-1]
            file_name = file_name[:-6]

            # print(f'graph files: {graph_files}')
            ### load graphs
            # if load_graphs:
            graph_file = open(file_path, 'rb')
            g = pickle.load(graph_file)
            data2 = data[data.folder_name == file_name].copy()
            data2.reset_index(inplace=True)
            for jdx, text in enumerate(data2.text):
                graphs_list.append([g, text, data2.loc[jdx,'score'], data2.loc[jdx,'name']])

            # graphs_list.append(g)
            pytorch_models.append(g.model)

    return graphs_list

if __name__ == '__main__':
    load_clone_ds(dataFrame='',
             load_path='', bimodal=False, num_nets=None, shuffle=False)