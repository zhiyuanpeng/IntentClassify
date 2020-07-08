# dense layer on top of the sentence embedding of bert
import torch
import torch.nn as nn
import transformers as ppb
from utils.torch_util import get_device


class BertSentClassify(nn.Module):
    """
    embedding layer consists of bert + char-lstm
    """

    def __init__(self, n_class, bert_model='bert-base-uncased'):
        super().__init__()
        # bert embedding layer
        self.bert = ppb.BertModel.from_pretrained(bert_model)
        # bert output 768 for each word
        self.bert_dim = 768
        self.dropout = nn.Dropout(p=0.5)
        self.device = get_device('duda')
        # convert to the dim=# of labels
        self.labeler = nn.Sequential(
            nn.ReLU(),
            # for Linear layer, initialization only need to denote input_dim, output_dim
            # the input data (N, *, input_dim) * is any other dimension we need, output (N, *, output_dim)
            # input_dim = lstm_hidden_dim*2 is the output_dim of self.lstm
            nn.Linear(768, 300),
            nn.ReLU(),
            nn.Linear(100, n_class)
        )

    def forward(self, sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices):
        # sentences (batch_size, max_sent_len)
        # sentence_length (batch_size)
        with torch.no_grad():
            last_hidden_states, _ = self.bert(sentences)
        # sentence features
        sents_rep = last_hidden_states[0][:, 0, :]
        final_out = self.labeler(sents_rep)
        return final_out


def main():
    pass


if __name__ == '__main__':
    main()
