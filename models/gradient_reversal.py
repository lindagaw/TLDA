"""GradientReversal model for ADDA."""

import torch.nn.functional as F
from torch import nn
from pytorch_revgrad import RevGrad


class GradientReversal(nn.Module):
    """GradientReversal gradientReversalEncoder model for ADDA."""

    def __init__(self):
        """Init GradientReversal gradientReversalEncoder."""
        super(GradientReversal, self).__init__()

        self.restored = False

        self.gradientReversalEncoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            RevGrad()
        )
        self.fc1_gradientReversal = nn.Linear(50 * 4 * 4, 500)
        self.fc2_gradientReversal = nn.Linear(500, 10)

    def forward(self, input):
        """Forward the GradientReversal."""
        conv_out = self.gradientReversalEncoder(input)
        feat = self.fc1_gradientReversal(conv_out.view(-1, 50 * 4 * 4))
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2_gradientReversal(out)
        return out
