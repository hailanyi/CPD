import torch.nn as nn
import torch

class TemporalConcatenation(nn.Module):
    def __init__(self, model_cfg, num_frames,input_channels):
        super(TemporalConcatenation, self).__init__()

        self.cfg=model_cfg
        self.num_frames=num_frames
        self.num_temporal_features=self.cfg.NUM_TEMPORAL_FEATURES
        self.input_channels=input_channels

        self.conv = nn.Conv2d(input_channels*self.num_frames, self.num_temporal_features, 3, padding=1,bias=False)
        self.bn = nn.BatchNorm2d(self.num_temporal_features,eps=1e-3, momentum=0.01)

    def forward(self,batch_dict):

        spatial_features=[]

        for i in range(self.num_frames):
            if i==0:
                spatial_features.append(batch_dict["spatial_features"])
            elif "spatial_features"+str(-i) in batch_dict:
                spatial_features.append(batch_dict["spatial_features"+str(-i)])

        spatial_features=torch.cat(spatial_features,1)

        temporal_features = self.conv(spatial_features)
        temporal_features = self.bn(temporal_features)

        batch_dict['temporal_features']=temporal_features

        return batch_dict