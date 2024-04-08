import torch
import torch.nn as nn
from torch.autograd import Variable


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())

class ConvLSTMbase(nn.Module):
    def __init__(self,num_features=128):
        super(ConvLSTMbase, self).__init__()
        self.input_channels=num_features
        self.hidden_channels=num_features

        self.lstm_cell = ConvLSTMCell(self.input_channels, self.hidden_channels, 3)

    def forward(self,inputs):
        h=None
        c=None

        keys=list(inputs.keys())
        keys.sort()

        for time_id in keys:

            x=inputs[time_id]

            if h is None and c is None:

                bsize, _, height, width = x.size()
                h, c = self.lstm_cell.init_hidden(batch_size=bsize, hidden=self.hidden_channels,
                                                 shape=(height, width))
            h, c = self.lstm_cell(x, h, c)

        return h


class ConvLSTM(nn.Module):
    def __init__(self, model_cfg, num_frames,input_channels):
        super(ConvLSTM, self).__init__()

        self.cfg=model_cfg
        self.num_frames=num_frames
        self.num_temporal_features=self.cfg.NUM_TEMPORAL_FEATURES
        self.input_channels=input_channels
        self.lstm=ConvLSTMbase(self.input_channels)

    def forward(self, batch_dict):

        spatial_features = {}

        for i in range(self.num_frames):
            if i == 0:
                spatial_features[0]=batch_dict["spatial_features"]
            elif "spatial_features" + str(-i) in batch_dict:
                spatial_features[-i]=batch_dict["spatial_features" + str(-i)]

        temporal_features = self.lstm(spatial_features)

        batch_dict['temporal_features'] = temporal_features
        return batch_dict
