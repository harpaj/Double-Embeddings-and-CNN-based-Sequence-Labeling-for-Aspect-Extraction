import torch
import torch.nn as tnn
import torch.nn.functional as F


class Model(tnn.Module):
    def __init__(self, gen_emb, domain_emb, num_classes=3, dropout=0.5):
        super(Model, self).__init__()
        self.gen_embedding = tnn.Embedding(gen_emb.shape[0], gen_emb.shape[1], padding_idx=0)
        self.gen_embedding.weight = tnn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = tnn.Embedding(domain_emb.shape[0], domain_emb.shape[1], padding_idx=0)
        self.domain_embedding.weight = tnn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)

        self.conv1 = tnn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 5, padding=2)
        self.conv2 = tnn.Conv1d(gen_emb.shape[1]+domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = tnn.Dropout(dropout)

        self.conv3 = tnn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = tnn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = tnn.Conv1d(256, 256, 5, padding=2)
        self.linear_ae = tnn.Linear(256, num_classes)

    def forward(self, x, x_len, x_mask, x_tag=None, testing=False):
        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(x)), dim=2)
        x_emb = self.dropout(x_emb).transpose(1, 2)
        x_conv = F.relu(torch.cat((self.conv1(x_emb), self.conv2(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = F.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = F.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = F.relu(self.conv5(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x_logit = self.linear_ae(x_conv)
        if testing:
            x_logit = x_logit.transpose(2, 0)
            score = F.log_softmax(x_logit).transpose(2, 0)
        else:
            x_logit = tnn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
            print(F.log_softmax(x_logit.data))
            print(x_tag.data)
            score = F.nll_loss(F.log_softmax(x_logit.data), x_tag.data)
        return score
