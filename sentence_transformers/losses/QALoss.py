import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

import sys
sys.path.append('..')
sys.path.append('...')
sys.path.append("....")
#from ppuda.ppuda.ghn.nn import GHN

class QALoss(nn.Module):
    #def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
    def __init__(self, model: SentenceTransformer=None, mam_head=None, mlm_head=None, archModel=None, qaModel=None, cross_encoder=False,
                 freeze_bert=True, loss_fct=nn.BCEWithLogitsLoss(reduction="mean"), main_loss_weight=1.0, mlm_loss_weight=1.0, mam_loss_weight=1.0):
        super(QALoss, self).__init__()
        self.model = model
        self.archModel = archModel
        self.qaModel = qaModel
        self.mlm_head=mlm_head
        self.mam_head = mam_head
        self.main_loss_weight = main_loss_weight
        self.mlm_loss_weight = mlm_loss_weight
        self.mam_loss_weight = mam_loss_weight

        self.cross_encoder = cross_encoder
        self.freeze_bert = freeze_bert
        self.loss_fct = loss_fct

    def compute_score_with_logits(self, logits, labels):
        logits = torch.max(logits, 1)[1].data  # argmax
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = one_hots * labels
        return scores

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, node_feats, shape_inds, edges, adjs, archs):
        #if self.model.module.archModel:
        if self.freeze_bert:
            with torch.no_grad():
                langmodel_output = self.model(sentence_features[0])
                embeddings_a_pooled = langmodel_output['sentence_embedding']
        else:
            langmodel_output = self.model(sentence_features[0])
            embeddings_a_pooled = langmodel_output['sentence_embedding'] # self.model: cross encoder

        # MLM
        mlm_loss = torch.tensor([0.0], requires_grad=True).cuda()
        if self.mlm_head is not None:
            token_embeddings = langmodel_output['token_embeddings']
            mlm_loss, mlm_logits = self.mlm_head(token_embeddings, sentence_features[0]['mlm_labels'])

        embeddings_b, embeddings_b_pooled = self.archModel(node_feats, shape_inds, edges, adjs)

        if self.cross_encoder:
            sentence_features[0]['input_ids'] = None
            sentence_features[0]['attention_mask'] = None
            if 'token_type_ids' in sentence_features[0]:
                #sa Bert has an issue when the model is called without embedding. We have to skip the embeder related parameters
                sentence_features[0]['token_type_ids'] = torch.zeros((embeddings_b.shape[:2])).int().cuda() #None
            sentence_features[0]['inputs_embeds'] = embeddings_b #embeddings_b_pooled.unsqueeze(dim=1)
            archmodel_output = self.model(sentence_features[0])
            embeddings_b_pooled = archmodel_output['sentence_embedding']

        predictions = self.qaModel(embeddings_a_pooled, embeddings_b_pooled)
        qa_loss = self.loss_fct(predictions, labels)

        # MAM
        mam_loss = torch.tensor([0.0], requires_grad=True).cuda()
        if self.mam_head is not None:
            if self.cross_encoder:
                token_embeddings = archmodel_output['token_embeddings']
            else:
                token_embeddings = embeddings_b
            mam_loss, mam_logits = self.mam_head(token_embeddings, sentence_features[0]['mam_labels'])

        #labels = None # called "target" in vilbert
        #labels = torch.ones([predictions.shape[0], 10], dtype=torch.float32).to(predictions.device)  # 64 classes, batch size = 10
        #labels = torch.randint(2, (predictions.shape[0], 10), dtype=torch.float32).to(predictions.device)
        # with binary labels: e.g., for 5 classes: [1,0,0,1,1]
        #labels = labels.view(-1)

        mlm_loss = mlm_loss.mean()
        mam_loss = mam_loss.mean()
        loss_value = self.mlm_loss_weight * mlm_loss + self.mam_loss_weight * mam_loss + self.main_loss_weight * qa_loss
        #.mean() * labels.size(1)
        # calcualte the score
        #batch_size = 8
        #batch_score = self.compute_score_with_logits(predictions, labels).sum() / float(batch_size)
        print('\nQA loss: %f' % (qa_loss.item()))
        print('MLM loss: %f' % (mlm_loss.item()))
        print('MAM loss: %f' % (mam_loss.item()))
        print('Total loss: %f' % (loss_value.item()))

        return qa_loss, mlm_loss, mam_loss, loss_value