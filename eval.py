# coding: utf-8
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import NERDataset
from utils.path_util import from_project_root
from utils.torch_util import f1_score

LABEL_LIST = ["O", "B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def evaluate(model, data_url):
    """
    evaluating end2end model on dataurl
    Args:
        model: trained end2end model
        data_url: url to test dataset for evaluating
    Returns:
        ret: dict of precision, recall, and f1
    """
    print("\nevaluating model on:", data_url, "\n")
    dataset = NERDataset(data_url, device=next(model.parameters()).device, bert_model="bert-base-uncased")
    loader = DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_func)
    # switch to eval mode
    model.eval()
    with torch.no_grad():
        sentence_true_list, sentence_pred_list = list(), list()
        for data, sentence_labels in loader:
            try:
                pred_sentence_labels = model.forward(*data, train=False)
                # pred_sentence_output (batch_size, n_classes, lengths[0])
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                continue
            sentence_lengths = data[1]
            for length, true_labels, pred_labels in zip(sentence_lengths, sentence_labels, pred_sentence_labels):
                for i in range(length):
                    true_labels_numpy = true_labels.cpu().numpy()
                    sentence_true_list.append(true_labels_numpy[i])
                    sentence_pred_list.append(pred_labels[i])
        precision, recall, f1 = f1_score(np.array(sentence_true_list), np.array(sentence_pred_list))
        print("Precision is %.4f, Recall is %.4f, F1 is %.4f" % (precision, recall, f1))
    return f1


def legal_num_labels(pred_labels, total_labels):
    pred_set = set()
    for i in range(len(pred_labels)):
        pred_set.add(pred_labels[i])
        if len(pred_set) == len(total_labels):
            return True
    return False


def main():
    model_url = from_project_root("data/model/lr0.000500_stop10_end2end_model_epoch52_0.722995.pt")
    test_url = from_project_root("data/Ali/Ali-test.iob2")
    model = torch.load(model_url)
    evaluate(model, test_url)
    pass


if __name__ == '__main__':
    main()
