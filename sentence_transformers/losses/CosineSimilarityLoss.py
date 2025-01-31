import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from ..SentenceTransformer import SentenceTransformer

import sys
sys.path.append('..')
sys.path.append('...')
sys.path.append("....")
#from ppuda.ppuda.ghn.nn import GHN

class CosineSimilarityLoss(nn.Module):
    """
    CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

    It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param model: SentenceTranformer model
    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

    Example::

            from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
                InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.CosineSimilarityLoss(model=model)


    """

    #def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
    def __init__(self, model: SentenceTransformer, mlm_head, archModel, mam_head, cross_encoder=False,
                 freeze_bert=True, loss_fct=nn.MSELoss(), cos_score_transformation=nn.Identity(), main_loss_weight=1.0, mlm_loss_weight=1.0, mam_loss_weight=1.0):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.mlm_head=mlm_head
        self.archModel = archModel
        self.mam_head = mam_head
        self.cross_encoder = cross_encoder
        self.freeze_bert = freeze_bert
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

        self.main_loss_weight = main_loss_weight
        self.mlm_loss_weight = mlm_loss_weight
        self.mam_loss_weight = mam_loss_weight

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

        #embeddings_b = self.archModel(graphs)
        if self.cross_encoder:
            sentence_features[0]['input_ids'] = None
            sentence_features[0]['attention_mask'] = None
            if 'token_type_ids' in sentence_features[0]:
                #sa Bert has an issue when the model is called without embedding. We have to skip the embeder related parameters
                sentence_features[0]['token_type_ids'] = torch.zeros((embeddings_b.shape[:2])).int().cuda() #None
                #self.model.max_seq_length = embeddings_b.shape[1]
                #self.model[0].auto_model.config.max_position_embeddings = embeddings_b.shape[1]
                #self.model[0].auto_model.embeddings.position_embeddings.num_embeddings =
                #self.model[0].auto_model.embeddings.position_embedding_type = ""
            sentence_features[0]['inputs_embeds'] = embeddings_b #embeddings_b_pooled.unsqueeze(dim=1)
            archmodel_output = self.model(sentence_features[0])
            embeddings_b_pooled = archmodel_output['sentence_embedding']

        # MAM
        mam_loss = torch.tensor([0.0], requires_grad=True).cuda()
        if self.mam_head is not None:
            if self.cross_encoder:
                token_embeddings = archmodel_output['token_embeddings']
            else:
                token_embeddings = embeddings_b # embeddings from archModel (arch encoder)
            # apply MAM to the Arch-encoder
            #mam_loss, mam_logits = self.mam_head(token_embeddings, sentence_features[0]['mam_labels'])
            # apply MAM to the cross-encoder
            mam_loss, mam_logits = self.mam_head(token_embeddings, sentence_features[0]['mam_labels'])

        output = self.cos_score_transformation(torch.cosine_similarity(embeddings_a_pooled, embeddings_b_pooled))

        sim_loss = self.loss_fct(output, labels.view(-1))
        # the following mean() is needed for multi-gpu training
        mlm_loss = mlm_loss.mean()
        mam_loss = mam_loss.mean()
        # ni for now, let's just ignore the nan loss values
        if torch.isnan(mlm_loss):
            mam_loss = torch.tensor([0.0], requires_grad=True).cuda()
        loss_value = self.mlm_loss_weight * mlm_loss + self.mam_loss_weight * mam_loss + self.main_loss_weight * sim_loss
        print('\nSim loss: %f' % (sim_loss.item()))
        print('MLM loss: %f' % (mlm_loss.item()))
        print('MAM loss: %f' % (mam_loss.item()))
        print('Total loss: %f' % (loss_value.item()))
        return sim_loss, mlm_loss, mam_loss, loss_value