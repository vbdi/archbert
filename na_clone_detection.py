import sys
sys.path.append('')
#sys.path.append('...')
sys.path.append('')
#sys.path.append('.')
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import streamlit as st
from ppuda.ppuda.deepnets1m.loader import DeepNets1M
import torch
from ppuda.ppuda.deepnets1m.graph import batch_graph_padding#, GraphBatch
from data.graph.load_graph import load_clone_ds as tvhf_load_clone_ds
import numpy as np
from tqdm.autonotebook import trange
from sklearn.metrics import f1_score,accuracy_score
import utils


def deepnet1m_batching_collate(batch):
    return batch

@st.cache(allow_output_mutation=True)
#@st.experimental_memo
def load_for_demo(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None):
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
        #graphs_list = tvhf_load(data_path+'/val_cd.csv', load_path=data_path+'/*.graph', load_from_path=True, unique_only=True, tokenizer_model_path=modelpath)
        graphs_list = tvhf_load_clone_ds(data_path + '/val_cd.csv', load_path=data_path + '/*.graph', num_nets=num_nets, shuffle=True)
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

        deepnet1m_loader, _ = DeepNets1M.loader(meta_batch_size=16, split='val', nets_dir=data_path,
                                         virtual_edges=1, num_nets=num_nets, large_images=False, max_node_size=max_node_size, max_edge_size=max_edge_size)
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
                texts.append(g.texts)
                labels.append(g.label)

    #if graph[0].edges:
    #    edges = torch.stack(edges).to('cuda')
    #else:
    #    edges = None

    all_archs_emb_pooled = []
    for start_index in trange(0, len(node_feats), batch_size):
        node_feats[start_index:start_index + batch_size], shape_inds[start_index:start_index + batch_size], _Adjs[start_index:start_index + batch_size] = batch_graph_padding(node_feats[start_index:start_index + batch_size], shape_inds[start_index:start_index + batch_size], _Adjs[start_index:start_index + batch_size], max_node_size)
        nf = torch.stack(node_feats[start_index:start_index + batch_size]).to('cuda')
        si = torch.stack(shape_inds[start_index:start_index + batch_size]).to('cuda')
        ad = torch.stack(_Adjs[start_index:start_index + batch_size]).to('cuda')

        archs_emb, archs_emb_pooled = archModel(node_feats=nf, shape_inds=si, edges=None, adjs=ad)
        if cross_encoder:
            archs_emb_pooled = langModel.encode(sentences=None, arch_embeds=archs_emb, batch_size=batch_size, convert_to_tensor=True)  # , normalize_embeddings=True)
        all_archs_emb_pooled.extend(archs_emb_pooled)

    return names, labels, all_archs_emb_pooled

def demo(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None):
    names, _, all_archs_emb_pooled = load_for_demo(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size,
         max_edge_size, hypernet)

    selected_arch_1 = st.selectbox('Input architecture 1:', names)
    selected_arch_index_1 = names.index(selected_arch_1)

    selected_arch_2 = st.selectbox('Input architecture 2:', names)
    selected_arch_index_2 = names.index(selected_arch_2)

    score = (util.cos_sim(all_archs_emb_pooled[selected_arch_index_1], all_archs_emb_pooled[selected_arch_index_2]))[0].cpu()

    return score

def load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None):
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
    if dataset=='tvhf':
        node_feats_1 = []
        shape_inds_1 = []
        _Adjs_1 = []
        names_1 = []
        node_feats_2 = []
        shape_inds_2 = []
        _Adjs_2 = []
        names_2 = []
        labels = []
        jaccard_scores = []
        eigen_distances = []
        cos_scores = []
        graphs_list = tvhf_load_clone_ds(data_path+'/val_cd.csv', load_path=data_path+'/*.graph', num_nets=num_nets, shuffle=False)
        for graph in graphs_list:
            node_feats_1.append(graph[0].node_feat)
            shape_inds_1.append(graph[0].shape_ind)
            _Adjs_1.append(graph[0]._Adj)
            names_1.append(graph[3])

            node_feats_2.append(graph[1].node_feat)
            shape_inds_2.append(graph[1].shape_ind)
            _Adjs_2.append(graph[1]._Adj)
            names_2.append(graph[4])

            labels.append(graph[2])

            ##### similarity computing using other methods
            G1 = graph[0]._nx_graph_from_adj()
            G2 = graph[1]._nx_graph_from_adj()
            jaccard_score, eigen_dis, cos_score = utils.get_scores(G1, G2, graph[0].node_feat, graph[1].node_feat, graph[0].shape_ind, graph[1].shape_ind, graph[0]._Adj, graph[1]._Adj)
            jaccard_scores.append(jaccard_score)
            #tacsim_scores.append(tacsim_score)
            eigen_distances.append(eigen_dis)
            cos_scores.append(cos_score)

            # edges.append(graph[0].edges)
    elif dataset=='autonet':
        node_feats = []
        shape_inds = []
        _Adjs = []
        names = []
        texts = []
        labels = []

        deepnet1m_loader, _ = DeepNets1M.loader(meta_batch_size=16, split='val', nets_dir=data_path,
                                         virtual_edges=1, num_nets=num_nets, large_images=False, max_node_size=max_node_size, max_edge_size=max_edge_size)
        deepnet1m_loader.collate_fn = deepnet1m_batching_collate
        deepnet1m_loader = iter(deepnet1m_loader)
        for (i,batch) in enumerate(deepnet1m_loader):
            #batch = next(deepnet1m_loader)
            if i == num_nets:
                break
            for g in batch:
                node_feat, shape_ind, _Adj = g.graph[0], g.graph[1], g.graph[3]
                node_feats.append(node_feat)
                shape_inds.append(shape_ind)
                _Adjs.append(_Adj)
                names.append(", ".join(g.unique_layers) + " - " + str(g.n_layers) + " layers - " + str(g.n_params) + " params")
                texts.append(g.texts)
                labels.append(g.label)

    all_archs_emb_pooled_1 = []
    all_archs_emb_pooled_2 = []

    for start_index in trange(0, len(node_feats_1), batch_size):
        node_feats_1[start_index:start_index + batch_size], shape_inds_1[start_index:start_index + batch_size], _Adjs_1[start_index:start_index + batch_size] = batch_graph_padding(node_feats_1[start_index:start_index + batch_size], shape_inds_1[start_index:start_index + batch_size], _Adjs_1[start_index:start_index + batch_size], max_node_size)
        nf_1 = torch.stack(node_feats_1[start_index:start_index + batch_size]).to('cuda')
        si_1 = torch.stack(shape_inds_1[start_index:start_index + batch_size]).to('cuda')
        ad_1 = torch.stack(_Adjs_1[start_index:start_index + batch_size]).to('cuda')
        node_feats_2[start_index:start_index + batch_size], shape_inds_2[start_index:start_index + batch_size], _Adjs_2[start_index:start_index + batch_size] = batch_graph_padding(node_feats_2[start_index:start_index + batch_size], shape_inds_2[start_index:start_index + batch_size], _Adjs_2[start_index:start_index + batch_size], max_node_size)
        nf_2 = torch.stack(node_feats_2[start_index:start_index + batch_size]).to('cuda')
        si_2 = torch.stack(shape_inds_2[start_index:start_index + batch_size]).to('cuda')
        ad_2 = torch.stack(_Adjs_2[start_index:start_index + batch_size]).to('cuda')

        archs_emb_1, archs_emb_pooled_1 = archModel(node_feats=nf_1, shape_inds=si_1, edges=None, adjs=ad_1)
        archs_emb_2, archs_emb_pooled_2 = archModel(node_feats=nf_2, shape_inds=si_2, edges=None, adjs=ad_2)
        if cross_encoder:
            archs_emb_pooled_1 = langModel.encode(sentences=None, arch_embeds=archs_emb_1, batch_size=batch_size, convert_to_tensor=True)  # , normalize_embeddings=True)
            archs_emb_pooled_2 = langModel.encode(sentences=None, arch_embeds=archs_emb_2, batch_size=batch_size, convert_to_tensor=True)  # , normalize_embeddings=True)
        else:
            archs_emb_pooled_1 = archs_emb_pooled_1.detach()
            archs_emb_pooled_2 = archs_emb_pooled_2.detach()
        all_archs_emb_pooled_1.extend(archs_emb_pooled_1)
        all_archs_emb_pooled_2.extend(archs_emb_pooled_2)


    return names_1, names_2, labels, all_archs_emb_pooled_1, all_archs_emb_pooled_2, jaccard_scores, eigen_distances, cos_scores

def validate(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None):
    names_1, names_2, labels, all_archs_emb_pooled_1, all_archs_emb_pooled_2, jaccard_scores, eigen_distances, cos_scores = load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size,
         max_edge_size, hypernet)

    accuracies = []
    all_scores = []
    all_labels = []

    for label, arch_emb_1, arch_emb_2 in zip(labels, all_archs_emb_pooled_1, all_archs_emb_pooled_2):
    #for label in zip(labels):#, all_archs_emb_pooled_1, all_archs_emb_pooled_2):
        scores = util.cos_sim(arch_emb_1, arch_emb_2)[:,0]
        #sa
        all_scores.append(torch.round(scores).cpu().numpy())
        all_labels.append(np.round(label))

    #sa
    #print("check the labels for the error")
    #print(all_labels)
    #print(all_scores)
    if len(all_scores)>0:
        all_scores = [item for sublist in all_scores for item in sublist]
        #print("all scores after changes")
        final_accuracy = accuracy_score(all_labels, all_scores)
        final_f1 = f1_score(all_labels, all_scores)
    else:
        final_accuracy = 0.0
        final_f1 = 0.0

    jacard_accuracy = accuracy_score(all_labels, np.round(jaccard_scores))
    jacard_f1 = f1_score(all_labels, np.round(jaccard_scores))

    #cos_accuracy = accuracy_score(all_labels, np.round(cos_scores))
    #cos_f1 = f1_score(all_labels, np.round(cos_scores))
    #print("cos_accuracy: " + str(cos_accuracy))
    #print("cos_f1: " + str(cos_f1))

    #print("------- eigen distances ---------")
    #print(eigen_distances)

    return final_accuracy, final_f1, jacard_accuracy, jacard_f1