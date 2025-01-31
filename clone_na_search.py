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
from data.graph.load_graph import load as tvhf_load
from tqdm.autonotebook import trange
import random
def deepnet1m_batching_collate(batch):
    '''
    node_feat = []
    shape_ind = []
    _Adj = []
    for g in batch:
        node_feat.append(g.graph[0])
        shape_ind.append(g.graph[1])
        _Adj.append(g.graph[3])
    '''
    return batch#node_feat, shape_ind, _Adj

@st.cache(allow_output_mutation=True)
#@st.experimental_memo
def load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None):
    if not langModel:
        langModel = SentenceTransformer(modelpath).cuda().eval()

    #st.write("INSIDE")
    node_feats = []
    shape_inds = []
    _Adjs = []
    edges = []
    names = []
    if dataset=='tvhf':
        graphs_list = tvhf_load( data_path+'/combined_neg_and_pos_datasets.csv', load_path=data_path+'/*.graph', load_from_path=True, unique_only=True, tokenizer_model_path=modelpath)
        #graphs_list = tvhf_load_clone_ds(data_path + '/val_cd.csv', load_path=data_path + '/*.graph', num_nets=num_nets, shuffle=True)
        for graph in graphs_list:
            names.append(graph[3])
            #graph[0] = graph_padding(graph[0], max_node_size, max_edge_size)
            node_feats.append(graph[0].node_feat)
            shape_inds.append(graph[0].shape_ind)
            _Adjs.append(graph[0]._Adj)
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
                if i==0:
                    print(g.texts)

    #if graph[0].edges:
    #    edges = torch.stack(edges).to('cuda')
    #else:
    edges = None

    if not archModel:
        archModel = torch.load(modelpath + '/GHN.pth').to(
            'cuda').eval()
        archModel.device = 'cuda'
        if not hypernet:
            archModel.gnn = None
        # archModel.layernorm = True

    #arch_emb, arch_emb_pooled = archModel(node_feats=node_feats, shape_inds=shape_inds, edges=edges, adjs=_Adjs)
    #if cross_encoder:
    #    arch_emb_pooled = langModel.encode(sentences=None, arch_embeds=arch_emb, batch_size=8, convert_to_tensor=True)  # , normalize_embeddings=True)

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
    all_archs_emb_pooled = torch.stack(all_archs_emb_pooled).to('cuda')

    return langModel, archModel, all_archs_emb_pooled, names, node_feats, shape_inds, _Adjs

def demo(modelpath, archModel, langModel, dataset, data_path, num_nets, batch_size, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None):
    langModel, archModel, archs_emb_pooled, names, node_feats, shape_inds, _Adjs = load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet)

    #### get arch_query embeddings ####
    selected_arch = st.selectbox('Input architecture:', names)
    arch_query_index = names.index(selected_arch)

    arch_query_node_feat = node_feats[arch_query_index].unsqueeze(0).to('cuda')
    arch_query_shape_ind = shape_inds[arch_query_index].unsqueeze(0).to('cuda')
    arch_query_Adj = _Adjs[arch_query_index].unsqueeze(0).to('cuda')
    arch_query_name = names[arch_query_index]

    arch_query_emb, arch_query_emb_pooled = archModel(node_feats=arch_query_node_feat, shape_inds=arch_query_shape_ind, edges=None, adjs=arch_query_Adj)
    if cross_encoder:
        arch_query_emb_pooled = langModel.encode(sentences=None, arch_embeds=arch_query_emb, batch_size=1, convert_to_tensor=True)  # , normalize_embeddings=True)
    ###################################

    # Compute dot score between query and all archs embeddings
    scores = (util.cos_sim(arch_query_emb_pooled, archs_emb_pooled))[0].cpu().tolist()
    doc_score_pairs = list(zip(names, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    return arch_query_name, doc_score_pairs

### To be done - waiting for be's clone ds
def validate(modelpath, archModel, langModel, dataset, data_path, num_nets, batch_size, cross_encoder, max_node_size, max_edge_size, hypernet, arch_query, checkpoint_epoch=None):
    ### load graphs, archModel, and compute the embeddings
    #names, arch_emb_pooled = load_archModel_for_validation(modelpath, archModel, langModel, cross_encoder, max_node_size, max_edge_size)
    ###

    langModel, archModel, archs_emb_pooled, names, node_feats, shape_inds, _Adjs = load_for_validation(modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, cross_encoder, max_node_size, max_edge_size, hypernet)

    #### get arch_query embeddings ####
    if arch_query is None:
        arch_query_index = random.randint(0, len(names))
    else:
        arch_query_index = arch_query
    arch_query_node_feat = node_feats[arch_query_index].unsqueeze(0).to('cuda')
    arch_query_shape_ind = shape_inds[arch_query_index].unsqueeze(0).to('cuda')
    arch_query_Adj = _Adjs[arch_query_index].unsqueeze(0).to('cuda')
    arch_query_name = names[arch_query_index]

    arch_query_emb, arch_query_emb_pooled = archModel(node_feats=arch_query_node_feat, shape_inds=arch_query_shape_ind, edges=None, adjs=arch_query_Adj)
    if cross_encoder:
        arch_query_emb_pooled = langModel.encode(sentences=None, arch_embeds=arch_query_emb, batch_size=1, convert_to_tensor=True)  # , normalize_embeddings=True)
    ###################################

    # Compute dot score between query and all archs embeddings
    scores = (util.cos_sim(arch_query_emb_pooled, archs_emb_pooled))[0].cpu().tolist()
    doc_score_pairs = list(zip(names, scores))
    doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    return arch_query_name, doc_score_pairs