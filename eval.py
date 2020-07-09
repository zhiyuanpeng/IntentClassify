# coding: utf-8
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import IntentDataset
from utils.path_util import from_project_root
from utils.torch_util import binary_p_r_f1

id_dict = {0: 'negative', 1: 'positive'}


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
    dataset = IntentDataset(data_url, device=next(model.parameters()).device, bert_model="bert-base-uncased")
    loader = DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_func)
    # switch to eval mode
    model.eval()
    total_pred_labels = []
    total_labels = []
    total_sentences = []
    with torch.no_grad():
        for sentences, masks, labels, sentences_list in loader:
            try:
                pred_labels_output = model.forward(sentences, masks)
                # argmax is for multi class
                # pred_labels = torch.argmax(pred_labels_output, dim=1)
                pred_labels = (pred_labels_output > 0.5).int()
            except RuntimeError:
                print("all 0 tags, no evaluating this epoch")
                continue
            labels_cpu = list(labels.cpu().numpy())
            pred_labels_cpu = list(pred_labels.view(-1).cpu().numpy())
            total_pred_labels += pred_labels_cpu
            total_labels += labels_cpu
            total_sentences += sentences_list
        write_predict_result(total_sentences, total_labels, total_pred_labels)
        # with open("./data/SST2/pred_result.txt", 'a+') as
        precision, recall, f1 = binary_p_r_f1(np.array(total_labels), np.array(total_pred_labels))
        print("Precision is %.4f, Recall is %.4f, F1 is %.4f" % (precision, recall, f1))
    return precision


def write_predict_result(total_sentences, total_labels, total_pred_labels):
    with open("./data/SST2/final_result.txt", "a+") as f:
        for sentence, real, pred in zip(total_sentences, total_labels, total_pred_labels):
            result = " ".join(sentence) + "\t" + id_dict[real] + "\t" + id_dict[pred] + "\n"
            f.write(result)


def main():
    model_url = from_project_root("data/model/lr0.000020_stop10_epoch5_0.937028.pt")
    test_url = from_project_root("data/SST2/test.txt")
    model = torch.load(model_url)
    evaluate(model, test_url)
    pass


if __name__ == '__main__':
    main()
