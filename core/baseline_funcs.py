"""Pre-train baseline for source dataset."""
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model

import os
import numpy as np
from sklearn.metrics import accuracy_score

def train_baseline(baseline, data_loader):
    """Train baseline for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    baseline.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(baseline.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            try:
                preds = baseline(images)
            except:
                preds = baseline(images[:,0,:,:].unsqueeze(1))
            loss = criterion(preds, labels)

            # optimize source baseline
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_baseline(baseline, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(
                baseline, "ADDA-source-baseline-{}.pt".format(epoch + 1))

    # # save final model
    save_model(baseline, "ADDA-source-baseline-final.pt")

    return baseline


def eval_baseline(baseline, data_loader):
    """Evaluate baseline for source domain."""
    # set eval state for Dropout and BN layers
    baseline.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = baseline(images)
        loss += criterion(preds, labels).data
        pred_cls = preds.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))

def get_distribution(tgt_data_loader):
    """Evaluate baseline for source domain."""

    vectors = []
    # evaluate network
    for (images, labels) in tgt_data_loader:
        images = make_variable(images, volatile=True).detach().cpu().numpy()
        labels = make_variable(labels).squeeze_().detach().cpu().numpy()
        for image in images:
            vectors.append(np.linalg.norm(image))
    
    mean = np.mean(vectors)
    inv = np.cov(vectors)

    mahalanobis = []

    for v in vectors:
        diff = v - mean
        m_dist = diff * inv * diff
        mahalanobis.append(m_dist)
    
    m_mean = np.mean(mahalanobis)
    m_std = np.std(mahalanobis)

    m_coeff = 0

    for coeff in range(0, 10000):
        upper = m_mean + m_std * coeff
        lower = m_mean - m_std * coeff

        if sum(1 for m in mahalanobis if lower < m and m < upper) > len(mahalanobis) * 0.8:
            print('found coefficient = ' + str(m_coeff))
            m_coeff = coeff
            break

    return mean, inv, m_mean, m_std, m_coeff

def is_in_distribution(sample, mean, inv, m_mean, m_std, m_coeff):
    upper = m_mean + m_std * m_coeff
    lower = m_mean - m_std * m_coeff

    sample = np.linalg.norm(sample)

    m = np.linalg.norm((sample - mean) * inv * (sample - mean))

    if lower < m and m < upper:
        return True
    else:
        return False

def eval_baseline_ood(baseline, tgt_data_loader, tgt_data_loader_eval):
    """Evaluate baseline for source domain."""

    mean, inv, m_mean, m_std, m_coeff = get_distribution(tgt_data_loader)
    # set eval state for Dropout and BN layers
    baseline.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    ys_true = []
    ys_pred = []

    # evaluate network
    for (images, labels) in tgt_data_loader_eval:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_().detach().cpu().numpy()
        predictions = baseline(images).detach().cpu().numpy()
        images = images.detach().cpu().numpy()

        for image, label, pred in zip(images, labels, predictions):

            if is_in_distribution(image, mean, inv, m_mean, m_std, 100000):
                ys_true.append(label)
                ys_pred.append(np.argmax(pred))
            else:
                continue

    acc = accuracy_score(ys_true, ys_pred)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))