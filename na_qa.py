import sys

#sys.path.append('../..')
#sys.path.append('....')
sys.path.append('')
#sys.path.append('.')
import torch
from sklearn.metrics import f1_score,accuracy_score

from ppuda.ppuda.deepnets1m.loader import DeepNets1M
from ppuda.ppuda.deepnets1m.graph import batch_graph_padding#, GraphBatch

from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.readers import InputExample
import streamlit as st

from data.graph.load_graph import load as tvhf_load
from torch.utils.data import DataLoader
import numpy as np
from tqdm.autonotebook import trange
import torch.nn as nn
import math
import os
import random
from ppuda.ppuda.deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
import copy

from mask_head_utils import mask_head

################### fine-tuning ####################
class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)

class ArchBERTQA(nn.Module):
    def __init__(self, hidden_size=768, num_answers=3129, dropout_prob=0.1):
        #super(VILBertForVLTasks, self).__init__(config)
        super().__init__()
        self.archbert_prediction = SimpleClassifier(hidden_size, hidden_size * 2, num_answers)
        #self.arch_logit = nn.Linear(hidden_size, 1)
        #self.linguisic_logit = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, pooled_output_text, pooled_output_arch):
        pooled_output = self.dropout(pooled_output_text * pooled_output_arch)
        prediction = self.archbert_prediction(pooled_output)
        return prediction

def load_qa_dataset_tvhf(split='train', data_path='', batch_size=128, num_nets=512, pos_only=False, max_node_size=512, max_edge_size=512, tokenizer_model_path=None):
    train_samples = []
    dev_samples = []
    test_samples = []

    #create_sample_pytorch_ds(save_paths='test_dataset/arch_no_param')
    graphs_list, new_vocab = tvhf_load(data_path+'/combined_neg_and_pos_datasets.csv',
                              load_path=data_path+'/*.graph', return_vocab=True, tokenizer_model_path=tokenizer_model_path)

    if not num_nets:
        num_nets = len(graphs_list)
    else:
        random.seed(100)
        random.shuffle(graphs_list)
    for graph in graphs_list[:num_nets]:
        if type(graph[1]) != float:
            # ignore negative samples for now!
            if (not pos_only) or (pos_only and float(graph[2])==1.0):
                graph_clone = copy.copy(graph[0])
                #inp_example = InputExample(texts=[graph[1]], graph=graph_clone, arch=None, label=float(graph[2]))

                # zero-padd the graph elements
                #graph_clone = graph_padding(graph_clone, max_node_size, max_edge_size)
                ### instead, we will do batch-padding in SentenceTransformers

                inp_example = InputExample(texts=[graph[1]], graph=[graph_clone.node_feat, graph_clone.shape_ind, graph_clone.edges, graph_clone._Adj], arch=None, label=[float(graph[2])])

                if split == 'dev':
                    dev_samples.append(inp_example)
                elif split == 'test':
                    test_samples.append(inp_example)
                else:
                    train_samples.append(inp_example)
        else:
            print('text not available!')

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
    #dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=16)
    #test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=16)

    return train_dataloader, None, None, new_vocab
    #return train_samples, dev_samples, test_samples

def load_qa_dataset_deepnet1m(split='train', meta_batch_size=8, max_batch_size=10, data_path='', num_nets=128, max_node_size=512, max_edge_size=512, mode='single'):
    #random.seed(100)
    graphs_queue, all_answers, new_vocab = DeepNets1M.loader(meta_batch_size=meta_batch_size, split=split,
                                     nets_dir=data_path,
                                     virtual_edges=1, num_nets=num_nets, large_images=False,
                                     max_node_size=max_node_size, max_edge_size=max_edge_size, dataset_type='qa', mode=mode)

    '''
    graphs = next(graphs_queue)
    for graph in graphs:
        print(graph.description)
        nets_args = graph.net_args
        net = Network(is_imagenet_input=False, num_classes=10, compress_params=False, **nets_args).eval()
        # delete all params - keep architecture only
        for param in net.parameters():
            param.data = torch.tensor(param.shape).float()#.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        graph.model = net
    '''
    # send "graphs" for training

    #print('DONE')
    return graphs_queue, all_answers, new_vocab

################### validation ####################

def deepnet1m_batching_collate(batch):
    return batch

@st.cache(allow_output_mutation=True)
#@st.experimental_memo
def load_for_validation(qa_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, qaModel, cross_encoder, max_node_size, max_edge_size, hypernet, mode='single', checkpoint_epoch=None):
    if not langModel:
        langModel = SentenceTransformer(qa_modelpath).cuda().eval()

    if not archModel:
        archModel = torch.load(qa_modelpath + '/GHN.pth').to(
            'cuda').eval()
        archModel.device = 'cuda'
        if not hypernet:
            archModel.gnn = None
        # archModel.layernorm = True

    if not qaModel:
        qaModel = torch.load(qa_modelpath + '/qaModel.pth').to('cuda').eval()

    #st.write("INSIDE")
    node_feats = []
    shape_inds = []
    _Adjs = []
    edges = []
    names = []
    texts = []
    labels = []
    if dataset=='tvhf':
        graphs_list = tvhf_load(
            data_path+'/combined_neg_and_pos_datasets.csv',
            load_path=data_path+'/*.graph', load_from_path=True, unique_only=True, tokenizer_model_path=qa_modelpath)
        for graph in graphs_list:
            names.append(graph[3])
            #graph[0] = graph_padding(graph[0], max_node_size, max_edge_size)
            ### instead, we will do batch-padding in SentenceTransformers
            node_feats.append(graph[0].node_feat)
            shape_inds.append(graph[0].shape_ind)
            _Adjs.append(graph[0]._Adj)
            texts.append(graph[1])
            labels.append(graph[2])
            # edges.append(graph[0].edges)
    elif dataset=='autonet':
        deepnet1m_loader, all_answers, _ = DeepNets1M.loader(meta_batch_size=batch_size, split='val', nets_dir=data_path,
                                         virtual_edges=1, num_nets=num_nets, large_images=False, max_node_size=max_node_size, max_edge_size=max_edge_size, dataset_type='qa', mode=mode)
        num_answers = len(all_answers)
        deepnet1m_loader.collate_fn = deepnet1m_batching_collate
        deepnet1m_loader = iter(deepnet1m_loader)
        for (i,batch) in enumerate(deepnet1m_loader):
            #batch = next(deepnet1m_loader)
            for g in batch:
                node_feat, shape_ind, _Adj = g.graph[0], g.graph[1], g.graph[3]
                node_feats.append(node_feat)
                shape_inds.append(shape_ind)
                _Adjs.append(_Adj)
                names.append(", ".join(g.unique_layers) + " - " + str(g.n_layers) + " layers - " + str(g.n_params) + " params")
                texts.append(g.texts) # there is only a single question
                labels.append(g.label) # to be added from be QA dataset

    #if graph[0].edges:
    #    edges = torch.stack(edges).to('cuda')
    #else:
    #    edges = None

    all_archs_emb_pooled = []
    all_text_batch = []
    all_label_batch = []
    for start_index in trange(0, len(node_feats), batch_size):
        node_feats[start_index:start_index + batch_size], shape_inds[start_index:start_index + batch_size], _Adjs[start_index:start_index + batch_size] = batch_graph_padding(node_feats[start_index:start_index + batch_size], shape_inds[start_index:start_index + batch_size], _Adjs[start_index:start_index + batch_size], max_node_size)
        nf = torch.stack(node_feats[start_index:start_index + batch_size]).to('cuda')
        si = torch.stack(shape_inds[start_index:start_index + batch_size]).to('cuda')
        ad = torch.stack(_Adjs[start_index:start_index + batch_size]).to('cuda')

        archs_emb, archs_emb_pooled = archModel(node_feats=nf, shape_inds=si, edges=None, adjs=ad)
        if cross_encoder:
            archs_emb_pooled = langModel.encode(sentences=None, arch_embeds=archs_emb, batch_size=batch_size, convert_to_tensor=True)  # , normalize_embeddings=True)
        else:
            archs_emb_pooled = archs_emb_pooled.detach()

        all_archs_emb_pooled.append(archs_emb_pooled)

        text_batch = texts[start_index:start_index + batch_size]
        all_text_batch.append(text_batch)
        label_batch = labels[start_index:start_index + batch_size]
        all_label_batch.append(label_batch)

        #############

    return langModel, qaModel, names, all_text_batch, all_label_batch, all_archs_emb_pooled, all_answers

def validate(qa_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, qaModel, cross_encoder, max_node_size, max_edge_size, hypernet, mode='single', checkpoint_epoch=None):
    langModel, qaModel, names, all_text_batch, all_label_batch, all_archs_emb_pooled, _ = load_for_validation(qa_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, qaModel,
                        cross_encoder, max_node_size, max_edge_size, hypernet, mode)

    batch_f1_scores = []
    batch_acc_scores = []
    all_scores = []
    all_labels = []
    for text_batch, label_batch, archs_emb_pooled in zip(all_text_batch, all_label_batch, all_archs_emb_pooled):
        for text, label, arch_emb in zip(text_batch, label_batch, archs_emb_pooled):
            query_emb = langModel.encode(sentences=text, batch_size=len(text), convert_to_tensor=True)
            predictions = qaModel(query_emb, arch_emb.repeat(len(text), 1))
            # the following is just an example - needs to be replaced by actual labels from be QA dataset
            # labels = torch.randint(2, (predictions.shape[0], 10), dtype=torch.float32).to(predictions.device)

            #pred = (torch.FloatTensor(label).to(predictions.device) * (torch.sigmoid(predictions).data > 0.5)).sum() / float(len(text))
            preds = torch.sigmoid(predictions).data > 0.5

            all_scores.extend(preds.to("cpu").to(torch.int).numpy())
            all_labels.extend(torch.IntTensor(label).to("cpu").numpy())

            #batch_f1_score = f1_score(torch.IntTensor(label).to("cpu").numpy(), preds.to("cpu").to(torch.int).numpy(), average="samples")
            #batch_acc_score = accuracy_score(torch.IntTensor(label).to("cpu").numpy(), preds.to("cpu").to(torch.int).numpy())

            #batch_score = losses.QALoss().compute_score_with_logits(predictions, torch.FloatTensor(label).to(predictions.device)).sum() / float(len(text)).cpu().numpy()

            #batch_f1_scores.append(batch_f1_score)
            #batch_acc_scores.append(batch_acc_score)

    #f1 = np.mean(batch_f1_scores)
    #acc = np.mean(batch_acc_scores)

    final_f1 = f1_score(all_labels, all_scores, average="samples")
    final_accuracy = accuracy_score(all_labels, all_scores)

    return final_accuracy, final_f1

def demo(qa_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, qaModel, cross_encoder, max_node_size, max_edge_size, hypernet, mode='single', checkpoint_epoch=None):
    langModel, qaModel, names, all_text_batch, all_label_batch, all_archs_emb_pooled, all_answers = load_for_validation(qa_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, qaModel,
                        cross_encoder, max_node_size, max_edge_size, hypernet, mode)

    selected_arch = st.selectbox('Input architecture:', names)
    selected_arch_index = names.index(selected_arch) + 1

    list_index = math.ceil(selected_arch_index/batch_size) - 1
    batch_index = selected_arch_index-(list_index*batch_size) - 1

    query = ""
    selected_label = None
    if st._is_running_with_streamlit:
        query = st.text_input("Enter Question:", "").lower()
        questions = all_text_batch[list_index][batch_index:batch_index+1][0]
        selected_question = st.selectbox('Choose Question:', [None] + questions)
        if selected_question:
            selected_question_index = questions.index(selected_question)
            selected_label = all_label_batch[list_index][batch_index:batch_index + 1][0][selected_question_index]
            query = selected_question

    predicted_answers = []
    actual_answers = []
    f1 = 0.0
    acc = 0.0
    if len(query.strip())>0:
        query_emb = langModel.encode(query, convert_to_tensor=True)

        predictions = qaModel(query_emb, all_archs_emb_pooled[list_index][batch_index:batch_index+1])
        preds = torch.sigmoid(predictions).data > 0.5
        preds_indices = (preds[0]).nonzero()[:,0].cpu().numpy()
        predicted_answers = np.array(all_answers)[preds_indices]

        # If we wanna compare with labels and report scores, we should provide a predefined list of questions for the user to choose 
        # includes many questions, each has a separate answer vector of size 56
        if selected_label:
            actual_indices = (torch.IntTensor(selected_label)).nonzero()[:,0].cpu().numpy()
            actual_answers = np.array(all_answers)[actual_indices]

            f1 = f1_score(torch.IntTensor([selected_label]).to("cpu").numpy(), preds.to("cpu").to(torch.int).numpy(), average="samples")
            acc = accuracy_score(torch.IntTensor([selected_label]).to("cpu").numpy(), preds.to("cpu").to(torch.int).numpy())

    return predicted_answers, actual_answers, acc, f1

    '''
    logits = torch.max(predictions, 0)[1].data  # argmax
    one_hots = torch.zeros(*predictions.shape[1]).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    '''

    #print('TBD')