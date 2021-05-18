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

        dist_src = src_detector(images).squeeze_()
        dist_tgt = tgt_detector(images).squeeze_()

        likely_class_src = torch.argmax(dist_src)
        likely_class_tgt = torch.argmax(dist_tgt)

        print(dist_src.shape)
        print(likely_class_src)

        '''
        if dist_src[likely_class_src] > dist_tgt[likely_class_tgt]:
            encoder = src_encoder
        elif dist_src[likely_class_src] < dist_tgt[likely_class_tgt]:
            encoder = tgt_encoder
        else:
            continue
        '''

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data[0]

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
