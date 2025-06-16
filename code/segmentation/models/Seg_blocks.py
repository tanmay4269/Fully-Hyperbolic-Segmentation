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
#       Hyperbolic (Lorentz)
#################################################
class H_Encoder(nn.Module):
    """ Implementation of a fully hyperbolic convolutional encoder for embedding an image.
    """
    def __init__(self, 
            num_layers,
            initial_filters,
            learn_curvature = False,
            curvature = 1.0
        ):
        super(H_Encoder, self).__init__()

        self.eps = 1e-5

        self.manifold = CustomLorentz(k=curvature, learnable=learn_curvature)
        self.learn_curvature = learn_curvature

        # Convolutional layers
        self.conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i==0:
                in_channels = 3+1
            else:
                in_channels = (initial_filters*(2**(i-1))) + 1
            out_channels = (initial_filters*(2**i)) + 1

            self.conv_layers.add_module("Conv_"+str(i), LConv2d_Block(
                manifold=self.manifold, 
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=2, 
                padding=1,
                bias=True,
                activation=torch.relu,
                normalization="batch_norm")
            )


    def forward(self, x):
        # project image pixels to hyperbolic space
        x = x.permute(0,2,3,1)
        # -> FROM HERE: CHANNEL LAST!!!
        x = F.pad(x, pad=(1,0), mode="constant", value=0)
        x = self.manifold.projx(x)

        features = []
        for layer in self.conv_layers:
            x = layer(x)
            features.append(x)
        return features


class H_Decoder(nn.Module):
    """ Implementation of a fully hyperbolic convolutional decoder for segmentation.

    Takes a latent vector of dimension zDim and generates a segmentation map.
    """
    def __init__(
            self, 
            num_layers, 
            num_classes,
            initial_filters, 
            learn_curvature = False,
            curvature = 1.0
        ):
        super(H_Decoder, self).__init__()

        self.manifold = CustomLorentz(k=curvature, learnable=learn_curvature)

        self.pred_dim = 64

        self.conv_layers = nn.Sequential()
        for i in range(num_layers-1):
            in_channels = int(initial_filters/(2**(i)))
            out_channels = int(initial_filters/(2**((i+1))))
            self.conv_layers.add_module("TrConv_"+str(i), LTransposedConv2d_Block(
                manifold=self.manifold, 
                in_channels=in_channels+1, 
                out_channels=out_channels+1, 
                kernel_size=4, 
                stride=2, 
                padding=1,
                bias=True,
                activation=torch.relu,
                normalization="batch_norm"
            ))

        self.final_conv = LConv2d_Block( 
            manifold=self.manifold, 
            in_channels=int(initial_filters/2**(num_layers-1))+1, 
            out_channels=self.pred_dim+1, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=True,
        )

        self.predictor = LorentzMLR(num_features=self.pred_dim+1, num_classes=num_classes, manifold=self.manifold)


    def forward(self, features):
        for i, feature in enumerate(features):
            if i == 0:
                x = self.conv_layers[i](feature)
            else:
                x = self.conv_layers[i](x + feature)
        x = self.final_conv(x)
        
        x = self.predictor(x)
        x = torch.sigmoid(x)

        x = x.permute(0,3,1,2)

        return x


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