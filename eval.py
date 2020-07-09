# coding: utf-8
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import IntentDataset
from utils.path_util import from_project_root
from utils.torch_util import binary_p_r_f1


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
    with torch.no_grad():
        for sentences, masks, labels in loader:
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
        precision, recall, f1 = binary_p_r_f1(np.array(total_labels), np.array(total_pred_labels))
        print("Precision is %.4f, Recall is %.4f, F1 is %.4f" % (precision, recall, f1))
    return precision


def main():
    model_url = from_project_root("data/model/lr0.000500_stop10_end2end_model_epoch52_0.722995.pt")
    test_url = from_project_root("data/Ali/Ali-test.iob2")
    model = torch.load(model_url)
    evaluate(model, test_url)
    pass


if __name__ == '__main__':
    main()
