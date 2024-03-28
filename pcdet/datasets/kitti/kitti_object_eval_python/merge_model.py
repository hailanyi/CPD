import torch
import os
import collections

def merge_model(path, begin = 31, end = 40):
    all_models = []
    for i in range(begin,end+1):
        name = 'checkpoint_epoch_'+str(i)+'.pth'
        this_path = os.path.join(path,name)
        loc_type = torch.device('cpu')
        checkpoint = torch.load(this_path, map_location=loc_type)
        model_state_disk = checkpoint['model_state']
        all_models.append(model_state_disk)

    weight_keys = list(all_models[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(all_models)):
            key_sum = key_sum + all_models[i][key]
        fed_state_dict[key] = key_sum / len(all_models)


    models = collections.OrderedDict()
    models['model_state'] = fed_state_dict
    torch.save(models, os.path.join(path, 'merged_model.pth'), _use_new_zipfile_serialization=False)



if __name__ == '__main__':
    merge_model('/home/asc01/projects/OpenProjects/OpenPCDet/output_semi_v1/models/waymo2kitti/casa/kitti/CasA-V-car-semi/default/ckpt' )







