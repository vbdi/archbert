import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

import sys
sys.path.append('..')
sys.path.append('...')
sys.path.append("....")
#from ppuda.ppuda.ghn.nn import GHN
import random

class LangDecLoss(nn.Module):
    #def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
    def __init__(self, model: SentenceTransformer=None, mam_head=None, mlm_head=None, main_loss_weight=1.0, mlm_loss_weight=1.0, mam_loss_weight=1.0, archModel=None, langDec=None, cross_encoder=False, freeze_bert=True, loss_fct=nn.CrossEntropyLoss(ignore_index=-1)):
        super(LangDecLoss, self).__init__()
        self.model = model
        self.archModel = archModel
        self.langDec = langDec
        self.cross_encoder = cross_encoder
        self.freeze_bert = freeze_bert
        self.loss_fct = loss_fct

        self.mlm_head=mlm_head
        self.mam_head = mam_head
        self.main_loss_weight = main_loss_weight
        self.mlm_loss_weight = mlm_loss_weight
        self.mam_loss_weight = mam_loss_weight

    def compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * labels
        return scores

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, node_feats, shape_inds, edges, adjs, archs):
        #if self.model.module.archModel:
        ####### language encoder ##### we dont need it here for arch2lang
        #if self.freeze_bert:
        #    with torch.no_grad():
        #        embeddings_a = self.model(sentence_features[0])['sentence_embedding']
        #else:
        #    embeddings_a = self.model(sentence_features[0])['sentence_embedding']
        ###### arch encoder ######
        embeddings_b, embeddings_b_pooled = self.archModel(node_feats, shape_inds, edges, adjs)

        source_ids = sentence_features[0]['input_ids']
        source_mask = sentence_features[0]['attention_mask']
        target_mask = source_mask  # for now source and target are the same!

        if self.cross_encoder:
            sentence_features[0]['input_ids'] = None
            sentence_features[0]['attention_mask'] = None
            if 'token_type_ids' in sentence_features[0]:
                sentence_features[0]['token_type_ids'] = torch.zeros((embeddings_b.shape[:2])).int().cuda() #None
            sentence_features[0]['inputs_embeds'] = embeddings_b #embeddings_b_pooled.unsqueeze(dim=1)
            embeddings_b = self.model(sentence_features[0])['token_embeddings']

        ##### language decoder #####
        # for now we put target ids in source ids
        dec_shift_logits, dec_shift_labels = self.langDec(source_ids_shape=node_feats.shape, target_ids=source_ids, encoder_output=embeddings_b, source_mask=source_mask)
        # decode and print softmax(dec_shift_logits) and dec_shift_labels

        ################ loss calculation ####################
        if target_mask is not None:
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            # Flatten the tokens
            main_loss = self.loss_fct(dec_shift_logits.view(-1, dec_shift_logits.size(-1))[active_loss], dec_shift_labels.view(-1)[active_loss])
        else:
            # Flatten the tokens
            main_loss = self.loss_fct(dec_shift_logits.view(-1, dec_shift_logits.size(-1)), dec_shift_labels.view(-1))

        # No MLM is needed for this loss.
        mlm_loss = torch.tensor([0.0], requires_grad=True).cuda()

        # MAM
        mam_loss = torch.tensor([0.0], requires_grad=True).cuda()
        if self.mam_head is not None:
            if self.cross_encoder:
                token_embeddings = embeddings_b
            mam_loss, mam_logits = self.mam_head(token_embeddings, sentence_features[0]['mam_labels'])

        main_loss = main_loss.mean()
        mam_loss = mam_loss.mean()

        # if multi-gpu
        loss_value = self.mam_loss_weight * mam_loss + self.main_loss_weight * main_loss.mean()
        #outputs = loss_value,loss_value*active_loss.sum(),active_loss.sum()
        #return outputs

        if random.randint(0,100)==50:
            target_text = self.model.module.tokenizer.decode(list(dec_shift_labels.cpu().numpy())[0], clean_up_tokenization_spaces=False)
            pred_text = self.model.module.tokenizer.decode(list(torch.max(dec_shift_logits, 2)[1].data.cpu().numpy())[0], clean_up_tokenization_spaces=False)
            print('sample target text: ' + target_text)
            print('sample pred text: ' + pred_text)
        '''
        predicted_text = []
        for pred in preds[0]:
            t = pred.cpu().numpy()
            t = list(t)
            if 0 in t:
                t = t[:t.index(0)]
            text = langModel.tokenizer.decode(t, clean_up_tokenization_spaces=False)
            predicted_text.append(text)
        '''

        print('\nDecoder loss: %f' % (main_loss.item()))
        print('MLM loss: %f' % (mlm_loss.item()))
        print('MAM loss: %f' % (mam_loss.item()))
        print('Total loss: %f' % (loss_value.item()))

        return main_loss, mlm_loss, mam_loss, loss_value