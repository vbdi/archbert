import gc
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
from transformers import AutoModel
import argparse
import pandas as pd
import torchvision
import os, glob
import sys
sys.setrecursionlimit(100000)

print(os.getcwd())

from create_archbert_ds import save_pytorch_archs

def tvhf_dataset_generator(path= './datasets/tvhf/', num_nets=10):

    stop_models = ['video.r3d_18', 'video.mc3_18',
                   'video.r2plus1d_18',
                   'quantization.shufflenet_v2_x2_0',
                   'quantization.shufflenet_v2_x1_5']

    init_df = pd.read_csv(f'{path}tvhf_models_list.csv', encoding="UTF-8")
    if 'Unnamed: 0' in init_df.columns:
        init_df.drop('Unnamed: 0', axis=1, inplace=True)

    # Extract all unique models in the data set for efficiency
    main_dict = {}

    for i in range(len(init_df)):
        model_name = init_df.loc[i,'name']
        if model_name in stop_models:
            continue
        if model_name not in main_dict:

            main_dict[model_name] = (init_df.loc[i,'folder_name'],init_df.loc[i,'model'],
                                     init_df.loc[i,'input'])
    count = 0
    # making graphs:
    for model_name in main_dict:
        count += 1
        if count > num_nets:
            break
        print(f'models: {count}')
        file_name, type, input_size = main_dict[model_name]

        if type == 'TV':
            model = (eval('torchvision.models.%s()' % model_name)).eval()
            model.expected_image_sz = int(input_size)

            # pytorch_models.append((model, n))
            to_save_paths = path + file_name


        else:
            try:
                model = AutoModel.from_pretrained(model_name)
                model.expected_image_sz = -1
                to_save_paths = path + file_name

            except Exception as e:

                print(f'was not able to load {model_name}.')

        gc.collect()
        save_pytorch_archs([(model, model_name)], [to_save_paths])
        print('done')

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description='TVHF Dataset Generator')
    parser.add_argument("--num_nets", nargs='?', type=int, const=None, help="num_nets")
    parser.add_argument('--path', default='autonet', type=str, help='dataset')
    args = parser.parse_args()
    tvhf_dataset_generator(path=args.path, num_nets=args.num_nets)