import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from typing import Union
import numpy as np
from torch.autograd import Variable
import cv2

class ConvGRUCell(nn.Module):
    def __init__(self,input_dim, hidden_dim, bias=True):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.padding = 1, 1
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=3,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=3,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size,height,width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """

        b,_,h,w=input_tensor.size()
        if h_cur is None:
            h_cur = self.init_hidden(b,h,w)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next



class ConvGRUbase(nn.Module):

    # --------------------------------------------------------------------------
    # Initialization
    # --------------------------------------------------------------------------

    def __init__(self, num_features=128):

        super(ConvGRUbase, self).__init__()

        self.GRUcell = ConvGRUCell(num_features,
                                     num_features,
                                     )

    # --------------------------------------------------------------------------
    # Processing
    # --------------------------------------------------------------------------

    def forward(self, time_feature_list):
        keys = list(time_feature_list.keys())
        keys.sort()
        previous_state=None

        features={}

        for time_id in keys:

            previous_state = self.GRUcell(time_feature_list[time_id],previous_state)
            features[time_id]=previous_state

        return features

class ConvGRU(nn.Module):
    def __init__(self, model_cfg, num_frames,input_channels):
        super(ConvGRU, self).__init__()

        self.cfg=model_cfg
        self.num_frames=num_frames
        self.num_temporal_features=self.cfg.NUM_TEMPORAL_FEATURES
        self.input_channels=input_channels
        self.gru=ConvGRUbase(self.input_channels)

    def forward(self, batch_dict):

        spatial_features = {}

        for i in range(self.num_frames):
            if i == 0:
                spatial_features[0] = batch_dict["spatial_features"]
            elif "spatial_features" + str(-i) in batch_dict:
                spatial_features[-i] = batch_dict["spatial_features" + str(-i)]

        temporal_features = self.gru(spatial_features)

        for key in temporal_features.keys():
            if key == 0:
                batch_dict['temporal_features'] = temporal_features[key]
            else:
                batch_dict['temporal_features'+str(key)] = temporal_features[key]

        return batch_dict
