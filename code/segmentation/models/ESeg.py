import torch
import torch.nn as nn
import torch.nn.functional as F

from .Seg_blocks import E_Encoder, E_Decoder


class ESeg(nn.Module):
    """ 
    Implementation of a Euclidean Segmentation model.

    Args:
        img_dim: dimensionality of input image (d x H x W)
        enc_layers: Number of encoder convolutional layers
        dec_layers: Number of decoder tranposed convolutional layers
        z_dim: Number of latent dimensions
        initial_filters: Number of output filters of first convolutional layer. Gets doubled with each conv. layer.
        latent_distr: ["euclidean","lorentz","poincare"]
        learn_curvature: Set if curvature of hyperbolic embedding space should be learnable
        embed_K: Initial curvature of hyperbolic embedding space
    """

    def __init__(self, 
            enc_layers, 
            dec_layers, 
            initial_filters, 
            out_dim,
            latent_distr = "euclidean", 
            learn_curvature = False,
            embed_K = 1.0,
        ):
        super(ESeg, self).__init__()

        self.latent_distr = latent_distr
        
        self.encoder = E_Encoder(
            enc_layers, 
            initial_filters
        )

        self.decoder = E_Decoder(
            dec_layers, 
            initial_filters*(2**(enc_layers-1)),
            out_dim
        )

    def forward(self, x):
        out = self.decoder(self.encoder(x))
        upsampled = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        return upsampled
        