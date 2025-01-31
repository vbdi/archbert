import torch.nn as nn
from transformers.activations import gelu

class mask_head(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=30552):
        #super(VILBertForVLTasks, self).__init__(config)
        super().__init__()
        self.vocab_transform = nn.Linear(hidden_size, hidden_size)
        self.vocab_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.vocab_projector = nn.Linear(hidden_size, vocab_size)
        self.mlm_loss_fct = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        prediction_logits = self.vocab_transform(embeddings)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))#.mean()

        output = (prediction_logits,)# + dlbrt_output[1:]

        return ((mlm_loss,) + output) if mlm_loss is not None else output
