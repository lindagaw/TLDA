"""Test script to classify target data."""
import numpy as np
import torch
import torch.nn as nn

from utils import make_variable


def eval_tgt(src_encoder, tgt_encoder, classifier, data_loader, src_detector, tgt_detector):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    src_encoder.eval()
    tgt_encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        torch.no_grad()

        dists_src = src_detector(images).squeeze_()
        dists_tgt = tgt_detector(images).squeeze_()

        src_or_tgt = []

        for dist_src, dist_tgt in zip(dists_src, dists_tgt):
            dist_src = torch.max(dist_src.squeeze())
            dist_tgt = torch.max(dist_src.squeeze())

            if dist_src < dist_tgt and dist_src < 400:
                src_or_tgt.append(1)
            else:
                src_or_tgt.append(0)

        preds_src_encoder = classifier(src_encoder(images))
        preds_tgt_encoder = classifier(tgt_encoder(images))

        preds = []

        for origin, pred_src_encoder, pred_tgt_encoder in zip (src_or_tgt, \
                                    preds_src_encoder, preds_tgt_encoder):
            pred_src_encoder = pred_src_encoder.cpu().detach().numpy()
            pred_tgt_encoder = pred_tgt_encoder.cpu().detach().numpy()
            if origin == 0:
                preds.append(pred_src_encoder)
            else:
                preds.append(pred_tgt_encoder)

        preds = torch.Tensor(np.asarray(preds)).cuda()

        loss += criterion(preds, labels).data

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
