import torch.nn as nn
import torch
from sentence_transformers import SentenceTransformer
from ppuda.ppuda.deepnets1m.loader import DeepNets1M
from ppuda.ppuda.deepnets1m.graph import batch_graph_padding#, GraphBatch
from sentence_transformers import losses
import os

import streamlit as st

from data.graph.load_graph import load as tvhf_load
from tqdm.autonotebook import trange
import torch.nn.functional as F
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
import random
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import numpy as np
from ppuda.ppuda.deepnets1m.genotypes import PRIMITIVES_DEEPNETS1M
import copy

from mask_head_utils import mask_head

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

class LangDecoder(nn.Module):
    def __init__(self, config, encoder_embeddings=None, beam_size=None, max_target_length=None, sos_id=None, eos_id=None):
        super(LangDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        if hasattr(config,'n_layers'):
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        else:
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_hidden_layers)
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)

        self.encoder_embeddings = encoder_embeddings
        self.torchscript = config.torchscript
        self.tie_weights()

        self.beam_size = beam_size
        self.max_target_length = max_target_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,self.encoder_embeddings.word_embeddings)

    def forward(self, source_ids_shape=None, encoder_output=None, target_ids=None, source_mask=None):
        encoder_output = encoder_output.permute([1, 0, 2]).contiguous()
        if target_ids is not None: # train
            tgt_embeddings = self.encoder_embeddings(target_ids).permute([1, 0, 2]).contiguous()
            ### if we wanna include the attention masks in the following - we need to pad input text to become the same size of max_node_size=i.e., 512
            if source_mask is not None:
                attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
                # ni pad the source-mask to have the same size as input graph size
                padded_source_mask = F.pad(input=source_mask, pad=(0, source_ids_shape[1] - source_mask.shape[1]), mode='constant', value=0)
                out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask, memory_key_padding_mask=(1 - padded_source_mask).bool())
            else:
                out = self.decoder(tgt_embeddings, encoder_output)#, tgt_mask=attn_mask, memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            return shift_logits, shift_labels
        else: # prediction
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids_shape[0]):
                context = encoder_output[:, i:i + 1]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                if source_mask is not None:
                    context_mask = source_mask[i:i + 1, :]
                    context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_target_length):
                    if beam.done():
                        break
                    tgt_embeddings = self.encoder_embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    if source_mask is not None:
                        attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                        out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask, memory_key_padding_mask=(1 - context_mask).bool())
                    else:
                        out = self.decoder(tgt_embeddings, context)
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                #del input_ids
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_target_length - len(p))).view(1, -1) for p in pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds

def load_dataset_tvhf(split='train', data_path='', batch_size=128, num_nets=512, pos_only=False, max_node_size=512, max_edge_size=512, tokenizer_model_path=""):
    train_samples = []
    dev_samples = []
    test_samples = []

    #create_sample_pytorch_ds(save_paths='test_dataset/arch_no_param')
    graphs_list, train_new_vocab = tvhf_load(data_path+'/train_main.csv', load_path=data_path+'/*.graph', return_vocab=True, tokenizer_model_path=tokenizer_model_path)
    _, val_new_vocab = tvhf_load(data_path + '/val_main.csv', load_path=data_path + '/*.graph', return_vocab=True, tokenizer_model_path=tokenizer_model_path)
    new_vocab = list(set(train_new_vocab + val_new_vocab))

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

def deepnet1m_batching_collate(batch):
    return batch

def get_text_predictions(langModel, preds, targets, top=0):
    metric = BLEUScore()
    # rougeLsum needs NLTK that does not work on Roma
    rouge = ROUGEScore(accumulate='best', rouge_keys=('rouge1', 'rouge2', 'rougeL', 'rougeLsum'))

    all_predicted_texts = []
    all_rouge_scores = []
    all_bleu_scores = []
    for pred, target in zip(preds,targets):
        predicted_text = []
        for p in pred:
            t = p.cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            text = langModel.tokenizer.decode(t, clean_up_tokenization_spaces=False)
            predicted_text.append(text)

        all_predicted_texts.append(predicted_text)

        # ni calculating BLEU score
        # preds = ['the cat is on the mat','the cat is on the mat']
        # target = [['there is a cat on the mat', 'a cat is on the mat'],['there is a cat on the mat', 'a cat is on the mat']]
        bleu_score = metric(predicted_text[top], [target])
        all_bleu_scores.append(bleu_score)

        # ni calculating ROUGE scores - a better option for us
        rouge_score = rouge(predicted_text[top], target)
        all_rouge_scores.append(rouge_score)

    return all_predicted_texts, all_bleu_scores, all_rouge_scores

@st.cache(allow_output_mutation=True)
#@st.experimental_memo
def load_for_validation(langDec_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel,
                        langDecModel, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None, validate=False, pos_only=True):
    #st.write("INSIDE")
    node_feats = []
    shape_inds = []
    _Adjs = []
    edges = []
    names = []
    texts = []
    labels = []
    # only need the following for architecture visualization
    graphs = []
    if dataset=='tvhf':
        graphs_list = tvhf_load(data_path+'/val_main.csv', load_path=data_path+'/*.graph', load_from_path=True, unique_only=True, tokenizer_model_path=langDec_modelpath)
        for graph in graphs_list:
            if pos_only:
                pos_indices = (np.array(graph[2]) == 1.0).nonzero()
                text = list(np.array(graph[1])[pos_indices])
            else:
                text = graph[1]
            if len(text)>0:
                texts.append(text)
                names.append(graph[3])
                ### instead, we will do batch-padding in SentenceTransformers
                #graph[0] = graph_padding(graph[0], max_node_size, max_edge_size)
                node_feats.append(graph[0].node_feat)
                shape_inds.append(graph[0].shape_ind)
                _Adjs.append(graph[0]._Adj)

                if not validate:
                    graphs.append(graph[0])

            #labels.append(graph[2])
            # edges.append(graph[0].edges)
    elif dataset=='autonet':
        deepnet1m_loader, _ = DeepNets1M.loader(meta_batch_size=batch_size, split='val', nets_dir=data_path, virtual_edges=1, num_nets=num_nets, large_images=False, max_node_size=max_node_size, max_edge_size=max_edge_size, dataset_type='arch2lang')
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
                g.texts = [text.lower() for text in g.texts]
                texts.append(g.texts)
                if not validate:
                    graphs.append(g.arch)

    if not langModel:
        langModel = SentenceTransformer(langDec_modelpath).cuda().eval()

    if not archModel:
        archModel = torch.load(langDec_modelpath + '/GHN.pth').to(
            'cuda').eval()
        archModel.device = 'cuda'
        if not hypernet:
            archModel.gnn = None
        # archModel.layernorm = True

    if not langDecModel:
        langDecModel = torch.load(langDec_modelpath + '/langDecModel.pth').to('cuda').eval()

    #if graph[0].edges:
    #    edges = torch.stack(edges).to('cuda')
    #else:
    #    edges = None

    all_text_preds = []
    all_bleu_scores = []
    all_rouge_scores = []
    all_archs_emb = []
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
            archs_emb = langModel.encode(sentences=None, arch_embeds=archs_emb, output_value="token_embeddings", batch_size=batch_size, convert_to_tensor=True)  # , normalize_embeddings=True)
            archs_emb = torch.stack(archs_emb)

        if validate:
            target_text_batch = texts[start_index:start_index + batch_size]
            langDecModel.beam_size = 10
            top = 0  # take the top (first) prediction of beam search
            preds = langDecModel(source_ids_shape=[archs_emb.shape[0], 1], target_ids=None, encoder_output=archs_emb, source_mask=None)
            text_predictions, bleu_scores, rouge_scores = get_text_predictions(langModel, preds, target_text_batch, top)
            all_bleu_scores.extend(bleu_scores)
            all_rouge_scores.extend(rouge_scores)
            all_text_preds.extend(text_predictions)

        all_archs_emb.extend(archs_emb)
        #all_archs_emb_pooled.append(archs_emb_pooled)

        #all_text_batch.append(text_batch)
        #label_batch = labels[start_index:start_index + batch_size]
        #all_label_batch.append(label_batch)

        #############
    if validate:
        return names, texts, all_text_preds, all_bleu_scores, all_rouge_scores
    else:
        return langModel, langDecModel, names, texts, all_archs_emb, graphs

def demo(langDec_modelpath, archModel, langModel, langDecModel, dataset, data_path, num_nets, batch_size, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None, pos_only=True, visualize=False):
    langModel,langDecModel, names, all_target_texts, all_archs_emb, graphs = load_for_validation(langDec_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, langDecModel, cross_encoder, max_node_size, max_edge_size, hypernet, validate=False, pos_only=pos_only)

    selected_arch = st.selectbox('Input architecture:', names)
    selected_arch_index = names.index(selected_arch)

    if visualize:
        plt_legends = graphs[selected_arch_index].visualize(node_size=100,vis_legend=True,with_labels=True,label_offset=5, font_size=12)
        plt_legends.savefig('plt_legends.png')
        plt = graphs[selected_arch_index].visualize(node_size=50)#, vis_legend=True, with_labels=True, label_offset=5,font_size=12)
        #fig_html = mpld3.fig_to_html(plt)
        #components.html(fig_html, height=600)
        st.pyplot(plt)

    langDecModel.beam_size = 10
    #langDecModel.max_target_length = 32
    preds = langDecModel(source_ids_shape=[1,1], target_ids=None, encoder_output=all_archs_emb[selected_arch_index].unsqueeze(0), source_mask=None)
    top = 0  # take the top (first) prediction of beam search
    predicted_text, bleu_score, rouge_score = get_text_predictions(langModel, preds, all_target_texts[selected_arch_index], top)

    return predicted_text[0][:top+1], all_target_texts[selected_arch_index], bleu_score[0], rouge_score[0]

def validate(langDec_modelpath, archModel, langModel, langDecModel, dataset, data_path, num_nets, batch_size, cross_encoder, max_node_size, max_edge_size, hypernet, checkpoint_epoch=None, pos_only=True):
    names, all_target_texts, all_text_preds, all_bleu_scores, all_rouge_scores = load_for_validation(langDec_modelpath, dataset, data_path, num_nets, batch_size, archModel, langModel, langDecModel, cross_encoder, max_node_size, max_edge_size, hypernet, validate=True, pos_only=pos_only)

    bleu_score = torch.stack(all_bleu_scores).mean()

    import pandas as pd
    df = pd.DataFrame(all_rouge_scores)
    rouge_score = dict(df.mean())

    return bleu_score, rouge_score
