"""Test script to classify target data."""

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
            dist_src = torch.max(dist_src)
            dist_tgt = torch.max(dist_src)
            if dist_src < dist_tgt:
                src_or_tgt.append(1)
            else:
                src_or_tgt.append(0)

        preds_src_encoder = classifier(src_encoder(images))
        preds_tgt_encoder = classifier(tgt_encoder(images))

        initialized = False

        for origin, pred_src_encoder, pred_tgt_encoder in zip (src_or_tgt, \
                                    preds_src_encoder, preds_tgt_encoder):
            if origin == 0:
                if initialized == False:
                    preds = pred_src_encoder
                    initialized = True
                else:
                    preds = torch.stack((preds, pred_src_encoder))
            else:
                if initialized == False:
                    preds = pred_tgt_encoder
                    initialized = True
                else:
                    preds = torch.stack((preds, pred_tgt_encoder))

        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
