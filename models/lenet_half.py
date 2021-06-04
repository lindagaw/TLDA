"""LeNetHalf model for ADDA."""

import torch.nn.functional as F
from torch import nn


class LeNetHalfEncoder(nn.Module):
    """LeNetHalf encoder model for ADDA."""

    def __init__(self):
        """Init LeNetHalf encoder."""
        super(LeNetHalfEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            #nn.Conv2d(20, 50, kernel_size=5),
            #nn.Dropout2d(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.ReLU()

        )
        self.fc1 = nn.Linear(50 * 4 * 4 * 3.6, 500)

    def forward(self, input):
        """Forward the LeNetHalf."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4 * 3.6))
        return feat


class LeNetHalfClassifier(nn.Module):
    """LeNetHalf classifier model for ADDA."""

    def __init__(self):
        """Init LeNetHalf encoder."""
        super(LeNetHalfClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNetHalf classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

CUDA = True if torch.cuda.is_available() else False


'''
MODELS
'''


def CORAL(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss/(4*d*d)

    return loss


class DeepCORAL(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepCORAL, self).__init__()
        self.LeNetHalfEncoder = LeNetHalfEncoder()
        self.LeNetHalfClassifier = LeNetHalfClassifier()
        self.fc = nn.Linear(500, 10)

        # initialize according to CORAL paper experiment
        self.fc.weight.data.normal_(0, 0.005)

    def forward(self, source, target):
        source = self.LeNetHalfEncoder(source)
        source = self.LeNetHalfClassifier(source)
        source = self.fc(source)

        target = self.LeNetHalfEncoder(target)
        target = self.Classifier(target)
        target = self.fc(target)
        return source, target
