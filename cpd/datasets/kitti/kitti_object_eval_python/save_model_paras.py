import torch
import os
import collections


name = 'checkpoint_epoch_31.pth'

loc_type = torch.device('cpu')
checkpoint = torch.load(name, map_location=loc_type)
fed_state_dict = checkpoint['model_state']
models = collections.OrderedDict()
models['model_state'] = fed_state_dict
torch.save(models, 'V2'+name, _use_new_zipfile_serialization=False)
