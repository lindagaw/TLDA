"""Pre-train gradientReversal for source dataset."""
import torch.nn as nn
import torch.optim as optim

import params
from utils import make_variable, save_model


def train_gradientReversal(gradientReversal, data_loader):
    """Train gradientReversal for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    gradientReversal.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(gradientReversal.parameters()),
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
                preds = gradientReversal(images)
            except:
                preds = gradientReversal(images[:,0,:,:].unsqueeze(1))
            loss = criterion(preds, labels)

            # optimize source gradientReversal
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
            eval_gradientReversal(gradientReversal, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(
                gradientReversal, "ADDA-source-gradientReversal-{}.pt".format(epoch + 1))

    # # save final model
    save_model(gradientReversal, "ADDA-source-gradientReversal-final.pt")

    return gradientReversal


def eval_gradientReversal(gradientReversal, data_loader):
    """Evaluate gradientReversal for source domain."""
    # set eval state for Dropout and BN layers
    gradientReversal.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = gradientReversal(images)
        loss += criterion(preds, labels).data
        pred_cls = preds.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
