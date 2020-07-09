import os
import torch
import joblib
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers as ppb

from utils.path_util import from_project_root, dirname

# LABEL_LIST = ['followup', 'initiator', 'noninitiator']
LABEL_LIST = ['0', '1']


class IntentDataset(Dataset):
    def __init__(self, data_url, device, bert_model):
        super().__init__()
        self.data_url = data_url
        self.bert_model = bert_model
        self.label_list = LABEL_LIST
        self.sentences, self.labels = load_raw_data(data_url)
        self.device = device

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)

    def collate_func(self, data_list):
        """
        for xx in collate_func(batch_size, sentences, labels)
        :param data_list:
        [([sentence1], label1),
        ([sentence2], label2),
        ...total batch size #]
        ([sentence], label) format
        (['Bayern', 'München', 'ist', 'wieder', 'alleiniger'], followup, initiator, or noninitiator)
        :return:
        """
        # sort the data_list according to the len of the sentence
        data_list = sorted(data_list, key=lambda tup: len(tup[0]), reverse=True)
        sentences_list, labels_list = zip(*data_list)  # un zip
        max_len = len(sentences_list[0])
        """
        sentence_tensors is tup(sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)
        sentences: [sentence1, sentence2, ...]
                   sentence1 = [word1, word2, ...] sentence1 is the longest sent in batch, other sents are padded
        """
        sentence_tensors, masks = gen_sentence_tensors(sentences_list, self.device, self.bert_model)
        # (sentences, sentence_lengths, sentence_words, sentence_word_lengths, sentence_word_indices)
        sentence_labels = list()
        for label in labels_list:
            # label is followup, initiator, or noninitiator
            sentence_labels.append(self.label_list.index(label))
        sentence_labels = torch.LongTensor(sentence_labels).to(self.device)
        return sentence_tensors, masks, sentence_labels, sentences_list


def gen_sentence_tensors(sentence_list, device, bert_model):
    """
    generate input tensors from sentence list
    :param sentence_list: [[w1, w2,...,wn](sent1), [w1, w2,...](sent2), ...] len(sentence_list) = batch_size
    :param device: torch device
    :return:
    sentences: [sentence1, sentence2, ...]
               sentence1 = [word1, word2, ...] sentence1 is the longest sent in batch, other sents are padded
    """
    sentences = list()
    masks = list()
    # initialize the tokenizer
    bert_tokenizer = ppb.BertTokenizer.from_pretrained(bert_model)
    for sent in sentence_list:
        # word to word id
        # for intent classification, we need to add the special tokens
        sentence = torch.LongTensor(bert_tokenizer.encode(sent, add_special_tokens=True)).to(device)
        shape = sentence.shape
        mask = [1 for i in range(list(sentence.shape)[0])]
        mask = torch.LongTensor(mask).to(device)
        sentences.append(sentence)
        masks.append(mask)

    # (batch_size)
    sentences = pad_sequence(sentences, batch_first=True).to(device)
    masks = pad_sequence(masks, batch_first=True).to(device)
    # (batch_size, max_sent_len)
    return sentences, masks


def load_raw_data(data_url, update=False):
    """
    load data into sentences and labels
    :param data_url: url of data file
    :param update: whether force to update
    :return: sentences, labels
    """
    # load from pickle
    save_url = data_url.replace('.txt', '.raw.pkl')
    if not update and os.path.exists(save_url):
        return joblib.load(save_url)

    sentences = list()
    labels = list()
    with open(data_url, 'r', encoding='utf-8') as iob_file:
        for line in iob_file:
            sentence, label = line.strip("\n").split("\t")
            sentences.append(sentence.split())
            labels.append(label)
    joblib.dump((sentences, labels), save_url)
    return sentences, labels


def main():
    data_urls = from_project_root("data/SST2/train.txt")
    load_raw_data(data_urls)
    pass


if __name__ == '__main__':
    main()
