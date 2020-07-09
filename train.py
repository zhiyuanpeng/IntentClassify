import os
import torch
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from utils.path_util import from_project_root
from utils.torch_util import get_device
from dataset import IntentDataset
from models.IntentClassify import BertSentClassify
from eval import evaluate
from utils.torch_util import set_random_seed
RANDOM_SEED = 233
set_random_seed(RANDOM_SEED)


EARLY_STOP = 10
LR = 2e-5
BATCH_SIZE = 32
MAX_GRAD_NORM = 5
N_TAGS = 9
LOG_PER_BATCH = 10

TRAIN_URL = from_project_root("data/SST2/train.txt")
DEV_URL = from_project_root("data/SST2/dev.txt")
TEST_URL = from_project_root("data/SST2/test.txt")


def train_end2end(n_epochs=3000,
                  train_url=TRAIN_URL,
                  test_url=TEST_URL,
                  dev_url=DEV_URL,
                  learning_rate=LR,
                  batch_size=BATCH_SIZE,
                  early_stop=EARLY_STOP,
                  clip_norm=MAX_GRAD_NORM,
                  device='cuda',
                  save_only_best=True
                  ):
    """
    Train deep exhaustive model, trained best model will be saved at 'data/model/'
    Args:
        n_epochs: number of epochs
        train_url: url to train data
        test_url: url to test data
        dev_url: url to dev data
        learning_rate: learning rate
        batch_size: batch_size
        early_stop: early stop for training
        clip_norm: whether to perform norm clipping, set to 0 if not need
        device: device for torch
        save_only_best: only save model of best performance
    """
    # print arguments
    arguments = json.dumps(vars(), indent=2)
    print("arguments", arguments)
    start_time = datetime.now()

    device = get_device(device)
    train_set = IntentDataset(train_url, device=device, bert_model='bert-base-uncased')
    train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False,
                              collate_fn=train_set.collate_func)
    # pass # of tags
    model = BertSentClassify(1)

    if device.type == 'cuda':
        print("using gpu,", torch.cuda.device_count(), "gpu(s) available!\n")
    else:
        print("using cpu\n")
    model = model.to(device)
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    cnt = 0
    max_f1, max_f1_epoch = 0, 0
    best_model_url = None
    for epoch in range(n_epochs):
        # switch to train mode
        model.train()
        batch_id = 0
        for sentences, masks, labels, sentences_list in train_loader:
            optimizer.zero_grad()
            pred_labels = model.forward(sentences, masks)
            loss = F.binary_cross_entropy(torch.sigmoid(pred_labels.view(-1).float()), labels.float())
            loss.backward()
            # gradient clipping
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            if batch_id % LOG_PER_BATCH == 0:
                print("epoch #%d, batch #%d, loss: %.12f, %s" %
                      (epoch, batch_id, loss.item(), datetime.now().strftime("%X")))
            batch_id += 1

        cnt += 1
        # evaluating model use development dataset or and additional test dataset
        f1 = evaluate(model, dev_url)
        if f1 > max_f1:
            max_f1, max_f1_epoch = f1, epoch
            if save_only_best and best_model_url:
                os.remove(best_model_url)
            best_model_url = from_project_root("data/model/lr%f_stop%d_epoch%d_%f.pt" % (LR, EARLY_STOP, epoch, f1))
            torch.save(model, best_model_url)
            cnt = 0

        print("maximum of f1 value: %.6f, in epoch #%d" % (max_f1, max_f1_epoch))
        print("training time:", str(datetime.now() - start_time).split('.')[0])
        print(datetime.now().strftime("%c\n"))

        if cnt >= early_stop > 0:
            break

    if test_url:
        best_model = torch.load(best_model_url)
        print("best model url:", best_model_url)
        print("evaluating on test dataset:", test_url)
        evaluate(best_model, test_url)

    print(arguments)


def main():
    train_end2end()
    pass


if __name__ == '__main__':
    main()
