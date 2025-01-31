import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
#####


import csv
import gzip
import os
from torch.utils.data import DataLoader
import copy
import sys
import random
import torch
import pickle
import os
import pandas as pd


sys.path.append('..')
from ppuda.ppuda.ghn.nn import GHN
from ppuda.ppuda.ghn.layers import ShapeEncoder
from ppuda.ppuda.utils import adjust_net
from ppuda.ppuda.deepnets1m.graph import Graph#, GraphBatch

def create_sample_pytorch_ds(save_paths=''):
    model1 = (eval('torchvision.models.%s()' % ('resnet18'))).eval()
    model1.expected_image_sz = 32

    pytorch_models = [[model1,'name1']]#, [model2,'name2'], [model3,'name3']]
    graphs_list = save_pytorch_archs(pytorch_models, save_paths=save_paths)

def save_pytorch_archs(pytorch_models, save_paths=None):
    graphs_list = []
    list_of_not_working_models = []

    if os.path.exists('Error_messages.csv'):
        error_df = pd.read_csv('Error_messages.csv')
        if 'Unnamed: 0' in error_df.columns:
            error_df.drop('Unnamed: 0', axis=1, inplace=True)


    else:
        new_data = {'name': [], 'error_msg': [], 'exc_type': [],
                    'fname': [], 'exc_tb': [], 'line_num': [], 'co_name': []}

        error_df = pd.DataFrame(new_data)

    err_df_idx = len(error_df)

    for idx, pytorch_model in enumerate(pytorch_models):
        try:
        #if True:
            pytorch_model,model_name = pytorch_model
            pytorch_model = adjust_net(pytorch_model, large_input=False)
            ### create and save graph ###
            #if create_graphs:
            g = Graph(pytorch_model, ve_cutoff=1)
            #g.visualize(node_size=50)
            ###
            # delete all weights, but store their shapes
            for param in pytorch_model.parameters():
                param.data = torch.tensor(param.shape).float()#.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            ### ni new items to be added to the graph
            _, params_map = GHN._map_net_params_v2(g)
            shape_ind = ShapeEncoder().eval().get_shape_info(g.n_nodes, params_map)
            g.shape_ind = shape_ind
            #print(shape_ind.shape)
            ###
            graphs_list.append(g)

            # save the arch only without params/weights
            if save_paths:
                save_path = save_paths[idx] if isinstance(save_paths, (list)) else save_paths + "_" + str(idx)
                graph_file = open(save_path + ".graph", 'wb')
                pickle.dump(g, graph_file)
                graph_file.close()
        except Exception as e:
            list_of_not_working_models.append(model_name)

            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno
            co_name = exception_traceback.tb_frame.f_code.co_name

            error_df.loc[err_df_idx, 'name'] = model_name
            error_df.loc[err_df_idx, 'error_msg'] = e
            error_df.loc[err_df_idx, 'exc_type'] = exception_type
            error_df.loc[err_df_idx, 'fname'] = filename
            error_df.loc[err_df_idx, 'exc_tb'] = exception_traceback
            error_df.loc[err_df_idx, 'line_num'] = line_number
            error_df.loc[err_df_idx, 'co_name'] = co_name

            err_df_idx += 1



    with open('problematic models list v2.txt', 'a') as f:
        for m in list_of_not_working_models:
            f.write(m+'\n')

    error_df.to_csv('Error_messages.csv')

    return graphs_list

def load_pytorch_archs(load_path, load_graphs=True):
    data = pd.read_csv('2.csv', encoding="ISO-8859-1")

    x = data.name.value_counts()
    x.to_csv('value_counts.csv')

    import glob
    graph_files = glob.glob(load_path)

    pytorch_models = []
    graphs_list = []
    with torch.no_grad():
        for idx, file_path in enumerate(graph_files):

            file_name = file_path.split('/')[-1]
            file_name = file_name[:-6]

            graph_file = open(file_path, 'rb')
            g = pickle.load(graph_file)
            data2 = data[data.folder_name == file_name].copy()
            count = 0
            for idx, label in enumerate(data2.label) :
                graphs_list.append([g,label,1.0])
                count += 1

            # graphs_list.append(g)
            pytorch_models.append(g.model)

            with open('counter.txt', 'a') as f:
                s = str(file_name)+" : "+str(count)+'\n'
                print(s)
                f.write(s)

    return graphs_list

def create_graph(pytorch_model, save_paths=None):
    #graphs_list = None
    graphs_list = []
    for idx, pytorch_model in enumerate(pytorch_model):
        g = Graph(pytorch_model, ve_cutoff=1)
        graphs_list.append(g)
        if save_paths:
            save_file = open(save_paths[idx], 'wb')
            pickle.dump(g, save_file)
    return graphs_list

if __name__ == '__main__':
    create_sample_pytorch_ds()