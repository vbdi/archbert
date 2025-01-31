import sys
sys.path.append('')
#sys.path.append('...')
sys.path.append('')
#sys.path.append('.')
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import streamlit as st
import torch
from ppuda.ppuda.deepnets1m.loader import DeepNets1M
from ppuda.ppuda.deepnets1m.graph import batch_graph_padding#, GraphBatch
from data.graph.load_graph import load as tvhf_load
import numpy as np
from tqdm.autonotebook import trange
from sklearn.metrics import f1_score,accuracy_score
#sa
from vis_utils import visualization_class # visualize_arch_text,gather_embedding, visualize_all
from vis_utils import AR_baseline_score
#from utils import AR_baseline_score
def deepnet1m_batching_collate(batch):
    return batch

@st.cache(allow_output_mutation=True)
#@st.experimental_memo
def load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder,
                        max_node_size, max_edge_size, hypernet, checkpoint_epoch=None,visualize=False):
    if not langModel:
        langModel = SentenceTransformer(modelpath).cuda().eval()

    if not archModel:
        archModel = torch.load(modelpath + '/GHN.pth').to(
            'cuda').eval()
        archModel.device = 'cuda'
        if not hypernet:
            archModel.gnn = None
        # archModel.layernorm = True

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
            data_path+'/val_main.csv',
            load_path=data_path+'/*.graph', load_from_path=True, unique_only=True, tokenizer_model_path=modelpath,num_nets=num_nets)
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
            if len(graph[1])==0:
                print('here')
    elif dataset=='autonet':

        deepnet1m_loader, _ = DeepNets1M.loader(meta_batch_size=16, split='val', nets_dir=data_path,
                                         virtual_edges=1, num_nets=num_nets, large_images=False, max_node_size=max_node_size, max_edge_size=max_edge_size)
        deepnet1m_loader.collate_fn = deepnet1m_batching_collate
        deepnet1m_loader = iter(deepnet1m_loader)
        for (i,batch) in enumerate(deepnet1m_loader):
            if i == num_nets:
                break
            #batch = next(deepnet1m_loader)
            for g in batch:
                node_feat, shape_ind, _Adj = g.graph[0], g.graph[1], g.graph[3]
                node_feats.append(node_feat)
                shape_inds.append(shape_ind)
                _Adjs.append(_Adj)
                names.append(", ".join(g.unique_layers) + " - " + str(g.n_layers) + " layers - " + str(g.n_params) + " params")
                texts.append(g.texts)
                labels.append(g.label)

    #if graph[0].edges:
    #    edges = torch.stack(edges).to('cuda')
    #else:
    #    edges = None

    all_archs_emb_pooled = []
    #sa
    all_arch_token_emb =[]
    for start_index in trange(0, len(node_feats), batch_size):
        node_feats[start_index:start_index + batch_size], shape_inds[start_index:start_index + batch_size], _Adjs[start_index:start_index + batch_size] = batch_graph_padding(node_feats[start_index:start_index + batch_size], shape_inds[start_index:start_index + batch_size], _Adjs[start_index:start_index + batch_size], max_node_size)
        nf = torch.stack(node_feats[start_index:start_index + batch_size]).to('cuda')
        si = torch.stack(shape_inds[start_index:start_index + batch_size]).to('cuda')
        ad = torch.stack(_Adjs[start_index:start_index + batch_size]).to('cuda')

        archs_emb, archs_emb_pooled = archModel(node_feats=nf, shape_inds=si, edges=None, adjs=ad)
        if cross_encoder:
            #sa: We need a non-pooled features. set output_value to get both the token embeddings and sentece embedding
            if visualize: #& (archs_emb.shape[1]<512)
                archs_token_emb = langModel.encode(sentences=None, arch_embeds=archs_emb, batch_size=batch_size,
                                                convert_to_tensor=True,output_value='token_embeddings') #output_value=None returns error
                all_arch_token_emb.append(archs_token_emb)

            archs_emb_pooled = langModel.encode(sentences=None, arch_embeds=archs_emb, batch_size=batch_size, convert_to_tensor=True)  # , normalize_embeddings=True)
        all_archs_emb_pooled.extend(archs_emb_pooled)

    #sa: return both the pooled and token_embeddings
    return langModel, names, texts, labels, all_archs_emb_pooled, all_arch_token_emb
    #return langModel, names, texts, labels, all_archs_emb_pooled

def demo(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, query, checkpoint_epoch=None):
    langModel, names, texts, labels, all_archs_emb_pooled, all_archs_emb = load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size,
         max_edge_size, hypernet)

    selected_arch = st.selectbox('Input architecture:', names)
    selected_arch_index = names.index(selected_arch)

    score = 0.0
    if len(query.strip())>0:
        query_emb = langModel.encode(query, convert_to_tensor=True)  # , normalize_embeddings=True)
        score = (util.cos_sim(query_emb, all_archs_emb_pooled[selected_arch_index]))[0].cpu()

    return score

def validate(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None,visualize=False,AR_baseline=True):
    #langModel, names, texts, labels, all_archs_emb_pooled = load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size,
    #     max_edge_size, hypernet)
    #sa
    langModel, names, texts, labels, all_archs_emb_pooled, all_archs_emb= load_for_validation(modelpath, dataset,data_path,
    num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size,max_edge_size, hypernet,visualize=visualize)

    accuracies = []
    f1_scores = []
    all_scores = []
    all_labels = []
    if AR_baseline:
        all_scores_baseline=[]
    # sa parameters for visualization
    counter = -1
    if visualize:
        visualization_mode = 1  # 0 for one arch some + so -, 1 for multiple arch multiple +
        vis_class = visualization_class(visualization_mode)
    for text, label, arch_emb in zip(texts, labels, all_archs_emb_pooled):
        # sa
        counter += 1
        if text is not None and len(text)>0:
            query_emb = langModel.encode(sentences=text, batch_size=len(text), convert_to_tensor=True)
            #pairwise_cos_sim
            scores = (torch.round(util.cos_sim(query_emb, arch_emb)[:,0])).cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(label)

            #sa for baseline and  embedding visuzalization
            if AR_baseline:
                all_scores_baseline = AR_baseline_score(text,names,counter,all_scores_baseline)
            if (visualize):
                if  1.0 in scores:
                    vis_class.main_visualize(scores,langModel,arch_emb,query_emb,text,names,counter)

            #accuracy = accuracy_score(scores, np.round(label)) #sum(1 for x, y in zip(scores, np.round(label)) if x == y) / float(len(scores))
            # accuracies.append(accuracy)
            #f1 = f1_score(scores, np.round(label))
            #f1_scores.append(f1)
        else:
            print(text)

    #total_accuracy = np.mean(accuracies)
    #total_f1 = np.mean(f1_scores)
    if AR_baseline:
        baseline_accuracy = accuracy_score(np.round(all_labels), np.array(all_scores_baseline))
        baseline_f1 = f1_score(np.round(all_labels), np.array(all_scores_baseline))
        print("Baseline accuracy:{0} and F1:{1}".format(baseline_accuracy,baseline_f1))
    final_accuracy = accuracy_score(np.round(all_labels), all_scores)
    final_f1 = f1_score(np.round(all_labels), all_scores)

    return final_accuracy, final_f1