"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim

import params
import numpy as np
from sklearn.metrics import accuracy_score
import math
from utils import make_variable
from utils import make_variable, save_model

import os
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def train_tgt_classifier(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
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
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
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
            eval_src_encoder(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-target-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-target-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-target-encoder-final.pt")
    save_model(classifier, "ADDA-target-classifier-final.pt")

    return encoder, classifier

def train_src_encoder(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
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
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
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
            eval_src_encoder(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "ADDA-source-encoder-final.pt")
    save_model(classifier, "ADDA-source-classifier-final.pt")

    return encoder, classifier

def train_tgt_encoder(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    #criterion = CORAL()
    criterion = nn.CrossEntropyLoss()

    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              params.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.data,
                              loss_tgt.data,
                              acc.data))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                "ADDA-critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                "ADDA-target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        "ADDA-critic-final.pt"))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        "ADDA-target-encoder-final.pt"))
    return tgt_encoder

def eval_src_encoder(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        preds = classifier(encoder(images))
        try:
            loss += criterion(preds, labels).data
        except:
            loss = criterion(preds, torch.max(labels, 1)[1]).data

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def eval_tgt_encoder(tgt_encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
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
        preds = classifier(tgt_encoder(images))
        loss += criterion(preds, labels).data

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()
    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))

def get_distribution(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, data_loader, which_data_loader):

    if os.path.isfile('snapshots//' + which_data_loader + '_mahalanobis_std.npy') and \
        os.path.isfile('snapshots//' + which_data_loader + '_mahalanobis_mean.npy') and \
        os.path.isfile('snapshots//' + which_data_loader + '_iv.npy') and \
        os.path.isfile('snapshots//' + which_data_loader + '_mean.npy'):

        print("Loading previously computed mahalanobis distances' mean and standard deviation ... ")
        mahalanobis_std = np.load('snapshots//' + which_data_loader + '_mahalanobis_std.npy')
        mahalanobis_mean = np.load('snapshots//' + which_data_loader + '_mahalanobis_mean.npy')
        iv = np.load('snapshots//' + which_data_loader + '_iv.npy')
        mean = np.load('snapshots//' + which_data_loader + '_mean.npy')

    else:

        print("Start calculating the mahalanobis distances' mean and standard deviation ... ")
        vectors = []
        for (images, labels) in data_loader:
            images = make_variable(images, volatile=True)
            labels = make_variable(labels).squeeze_()
            torch.no_grad()
            src_preds = src_classifier(src_encoder(images)).detach().cpu().numpy()
            tgt_preds = tgt_classifier(tgt_encoder(images)).detach().cpu().numpy()
            critic_at_src = critic(src_encoder(images)).detach().cpu().numpy()
            critic_at_tgt = critic(tgt_encoder(images)).detach().cpu().numpy()
            for image, label, src_pred, tgt_pred, src_critic, tgt_critic \
                            in zip(images, labels, src_preds, tgt_preds, critic_at_src, critic_at_tgt):
                vectors.append(np.linalg.norm(src_critic.tolist() + tgt_critic.tolist()))
                print('processing vector ' + str(src_critic.tolist() + tgt_critic.tolist()))

        mean = np.asarray(vectors).mean(axis=0)
        cov = np.cov(vectors)
        try:
            iv = np.linalg.inv(cov)
        except:
            iv = cov
        mahalanobis = np.asarray([distance.mahalanobis(v, mean, iv) for v in vectors])
        mahalanobis_mean = np.mean(mahalanobis)
        mahalanobis_std = np.std(mahalanobis)
        np.save('snapshots//' + which_data_loader + '_mahalanobis_mean.npy', mahalanobis_mean)
        np.save('snapshots//' + which_data_loader + '_mahalanobis_std.npy', mahalanobis_std)
        np.save('snapshots//' + which_data_loader + '_iv.npy', iv)
        np.save('snapshots//' + which_data_loader + '_mean.npy', mean)

    print("Finished obtaining the mahalanobis distances' mean and standard deviation on " + which_data_loader)
    return mahalanobis_mean, mahalanobis_std, iv, mean

def is_in_distribution(vector, mahalanobis_mean, mahalanobis_std, mean, iv):
    upper_coefficient = 0.1
    lower_coefficient = 0.1

    upper = mahalanobis_mean + upper_coefficient * mahalanobis_std
    lower = mahalanobis_mean - lower_coefficient * mahalanobis_std

    mahalanobis = distance.mahalanobis(vector, mean, iv)

    if lower < mahalanobis and mahalanobis < upper:
        return True
    else:
        return False



def eval_ADDA(src_encoder, tgt_encoder, src_classifier, tgt_classifier, critic, data_loader):

    src_mahalanobis_std = np.load('snapshots//' + 'src' + '_mahalanobis_std.npy')
    src_mahalanobis_mean = np.load('snapshots//' + 'src' + '_mahalanobis_mean.npy')
    src_iv = np.load('snapshots//' + 'src' + '_iv.npy')
    src_mean = np.load('snapshots//' + 'src' + '_mean.npy')

    tgt_mahalanobis_std = np.load('snapshots//' + 'tgt' + '_mahalanobis_std.npy')
    tgt_mahalanobis_mean = np.load('snapshots//' + 'tgt' + '_mahalanobis_mean.npy')
    tgt_iv = np.load('snapshots//' + 'tgt' + '_iv.npy')
    tgt_mean = np.load('snapshots//' + 'tgt' + '_mean.npy')

    """Evaluation for target encoder by source classifier on target dataset."""
    tgt_encoder.eval()
    src_encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    # set loss function
    criterion = nn.CrossEntropyLoss()
    # evaluate network

    y_trues = []
    y_preds = []

    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()
        torch.no_grad()

        src_preds = src_classifier(src_encoder(images)).detach().cpu().numpy()
        tgt_preds = tgt_classifier(tgt_encoder(images)).detach().cpu().numpy()
        critic_at_src = critic(src_encoder(images)).detach().cpu().numpy()
        critic_at_tgt = critic(tgt_encoder(images)).detach().cpu().numpy()

        for image, label, src_pred, tgt_pred, src_critic, tgt_critic \
                        in zip(images, labels, src_preds, tgt_preds, critic_at_src, critic_at_tgt):

            vector = np.linalg.norm(src_critic.tolist() + tgt_critic.tolist())

            # ouf of distribution:
            if not is_in_distribution(vector, tgt_mahalanobis_mean, tgt_mahalanobis_std, tgt_mean, tgt_iv) \
                and not is_in_distribution(vector, src_mahalanobis_mean, src_mahalanobis_std, src_mean, src_iv):
                continue
            # if in distribution which the target:
            elif is_in_distribution(vector, tgt_mahalanobis_mean, tgt_mahalanobis_std, tgt_mean, tgt_iv):
                y_pred = np.argmax(tgt_pred)
            else:
                y_pred = np.argmax(src_pred)

            #y_pred = np.argmax(tgt_pred)
            y_preds.append(y_pred)
            y_trues.append(label.detach().cpu().numpy())


    print("Avg Accuracy = {:2%}".format(accuracy_score(y_true=y_trues, y_pred=y_preds)))


def eval_tgt_with_probe(encoder, critic, src_classifier, tgt_classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    src_classifier.eval()
    tgt_classifier.eval()
    # init loss and accuracy
    loss = 0.0
    acc = 0.0
    f1 = 0.0

    ys_pred = []
    ys_true = []
    # set loss function
    criterion = nn.CrossEntropyLoss()
    flag = False
    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        probeds = critic(encoder(images))

        for image, label, probed in zip(images, labels, probeds):
            if torch.argmax(probed) == 1:
                pred = torch.argmax(src_classifier(encoder(torch.unsqueeze(image, 0)))).detach().cpu().numpy()
            else:
                pred = torch.argmax(tgt_classifier(encoder(torch.unsqueeze(image, 0)))).detach().cpu().numpy()

        ys_pred.append(np.squeeze(pred))
        ys_true.append(np.squeeze(label.detach().cpu().numpy()))

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)
    #f1 /= len(data_loader.dataset)
    print("Avg Accuracy = {:2%}".format(accuracy_score(y_true=y_trues, y_pred=y_preds)))
