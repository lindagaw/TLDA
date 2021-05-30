"""Pre-train coral for source dataset."""
import torch.nn as nn
import torch.optim as optim
import torch
import params
from utils import make_variable, save_model


def train_coral(coral, data_loader):
    """Train coral for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    coral.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(coral.parameters()),
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
                preds = coral(images)
            except:
                preds = coral(images[:,0,:,:].unsqueeze(1))
            loss = CORAL(preds, labels)

            # optimize source coral
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
            eval_coral(coral, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(
                coral, "ADDA-source-coral-{}.pt".format(epoch + 1))

    # # save final model
    save_model(coral, "ADDA-source-coral-final.pt")

    return coral


def eval_coral(coral, data_loader):
    """Evaluate coral for source domain."""
    # set eval state for Dropout and BN layers
    coral.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    #criterion = CORAL()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = coral(images)
        loss += CORAL(preds, labels).data
        pred_cls = preds.data.max(1)[1]

        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(torch.Tensor.float(target), 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCORAL, self).__init__()
        self.LeNetEncoder = LeNetEncoder()
        self.LeNetClassifier = LeNetClassifier()
        self.fc = nn.Linear(500, 10)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.LeNetEncoder(source)
        source = self.LeNetClassifier(source)
        source = self.fc(source)

        target = self.LeNetEncoder(target)
        target = self.Classifier(target)
        target = self.fc(target)
        return source, target
