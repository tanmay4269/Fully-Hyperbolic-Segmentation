import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Seg_blocks import *

class LSeg(nn.Module):
    """ Implementation of a fully hyperbolic segmentation model.

    Args:
        enc_layers: Number of encoder convolutional layers
        dec_layers: Number of decoder tranposed convolutional layers
        num_classes: Number of classes
        initial_filters: Number of output filters of first convolutional layer. Gets doubled with each conv. layer.
        learn_curvature: If curvature of hyperbolic space should be learnable
        enc_K: Encoder curvature
        dec_K: Decoder curvature
    """

    def __init__(self, 
        enc_layers, 
        dec_layers, 
        num_classes, 
        initial_filters,
        learn_curvature = False,
        enc_K = 1.0,
        dec_K = 1.0
    ):
        super(LSeg, self).__init__()

        self.encoder = H_Encoder(
            enc_layers, 
            initial_filters, 
            learn_curvature, 
            enc_K
        )

        self.decoder = H_Decoder(
            dec_layers, 
            num_classes, 
            initial_filters, 
            learn_curvature, 
            dec_K
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        return out

    def loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, ignore_index=255)