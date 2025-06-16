import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Flatten, Sequential

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers import LorentzMLR
from lib.lorentz.distributions import LorentzWrappedNormal
from lib.lorentz.blocks.layer_blocks import LFC_Block, LConv2d_Block, LTransposedConv2d_Block

from lib.geoopt.manifolds.stereographic import PoincareBall
from lib.poincare.distributions import PoincareWrappedNormal

from lib.Euclidean.blocks.layer_blocks import FC_Block, Conv2d_Block, TransposedConv2d_Block


#################################################
#       Euclidean
#################################################
class E_Encoder(nn.Module):
    """ Implementation of a convolutional encoder for embedding an image.
    """
    def __init__(
        self, 
        num_layers,
        initial_filters
    ):
        super(E_Encoder, self).__init__()
        self.eps = 1e-5
    
        # Convolutional layers
        self.conv_layers = Sequential()
        for i in range(num_layers):
            if i==0:
                in_channels = 3
            else:
                in_channels = (initial_filters*(2**(i-1)))
            out_channels = (initial_filters*(2**i))

            self.conv_layers.add_module("Conv_"+str(i), Conv2d_Block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1, 
                activation=torch.relu, 
                bias=True,
                normalization="batch_norm"
            ))

    def forward(self, x):
        features = []
        for layer in self.conv_layers:
            x = layer(x)
            features.append(x)
        return features

class E_Decoder(nn.Module):
    """ Implementation of a convolutional decoder for image prediction.

    Takes a latent vector of dimension zDim and generates an image.
    """
    def __init__(
        self, 
        num_layers, 
        initial_filters,
        out_dim,
    ):
        super(E_Decoder, self).__init__()

        # Layers
        self.conv_layers = Sequential()
        for i in range(num_layers-1):
            in_channels = int(initial_filters/(2**(i)))
            out_channels = int(initial_filters/(2**((i+1))))
            self.conv_layers.add_module("TrConv_"+str(i), TransposedConv2d_Block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=4, 
                stride=2, 
                padding=1, 
                activation=torch.relu, 
                bias=True,
                normalization="batch_norm"
            ))
        
        self.conv_layers.add_module("FinalConv", Conv2d_Block(
            in_channels=int(initial_filters/2**(num_layers-1)), 
            out_channels=out_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            activation=torch.relu,
            bias=True,
            normalization="batch_norm"
        ))

    def forward(self, features):
        for i, feature in enumerate(features[::-1]):
            if i == 0:
                x = self.conv_layers[i](feature)
            else:
                x = self.conv_layers[i](x + feature)
        return x