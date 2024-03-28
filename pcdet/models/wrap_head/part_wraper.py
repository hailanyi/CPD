import numpy as np
import torch
import torch.nn as nn
from functools import partial
from ...utils import box_coder_utils, common_utils, loss_utils

def gen_sample_grid(box, window_size=(4, 7), grid_offsets=(0, 0), spatial_scale=1.):
    N = box.shape[0]
    win = window_size[0] * window_size[1]
    xg, yg, wg, lg, rg = torch.split(box, 1, dim=-1)

    xg = xg.unsqueeze_(-1).expand(N, *window_size)
    yg = yg.unsqueeze_(-1).expand(N, *window_size)
    rg = rg.unsqueeze_(-1).expand(N, *window_size)

    cosTheta = torch.cos(rg)
    sinTheta = torch.sin(rg)

    xx = torch.linspace(-.5, .5, window_size[0]).type_as(box).view(1, -1) * wg
    yy = torch.linspace(-.5, .5, window_size[1]).type_as(box).view(1, -1) * lg

    xx = xx.unsqueeze_(-1).expand(N, *window_size)
    yy = yy.unsqueeze_(1).expand(N, *window_size)

    x = (xx * cosTheta + yy * sinTheta + xg)
    y = (yy * cosTheta - xx * sinTheta + yg)

    x = (x.permute(1, 2, 0).contiguous() + grid_offsets[0]) * spatial_scale
    y = (y.permute(1, 2, 0).contiguous() + grid_offsets[1]) * spatial_scale

    return x.view(win, -1), y.view(win, -1)

def bilinear_interpolate_torch_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W


    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)#28,K,1,1
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = torch.cat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / (W - 1))  # normalize to between  0 and 1

    samples[:, :, :, 1] = (samples[:, :, :, 1] / (H - 1))  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1  #28,K,1,2

    return torch.nn.functional.grid_sample(image, samples)

class PartWraper(nn.Module):
    def __init__(self, model_cfg, num_frames, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__()

        self.in_c = input_channels
        self.model_cfg = model_cfg
        self.num_frames = num_frames
        self.num_class = num_class
        self.class_names = class_names
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        self.grid_offsets = self.model_cfg.GRID_OFFSETS
        self.featmap_stride = self.model_cfg.FEATMAP_STRIDE
        self.num_parts = self.model_cfg.NUM_PARTS[0]*self.model_cfg.NUM_PARTS[1]
        self.win_size = self.model_cfg.NUM_PARTS
        self.train_score = self.model_cfg.TRAIN_SCORE

        self.out_c = self.num_parts*self.num_class

        self.convs_1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.in_c, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_c, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convs_conf = nn.Sequential(
            nn.Conv2d(self.in_c, self.out_c, 1, 1, padding=0, bias=False),
        )

        self.gen_grid_fn = partial(gen_sample_grid, grid_offsets=self.grid_offsets, spatial_scale=1 / self.featmap_stride)

        self.forward_dict={}

        self.loss_fn = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)


    def get_loss(self):

        confi_targets=self.forward_dict["confi_targets"]
        confi_preds=self.forward_dict["confi_preds"]

        new_confi_targets=torch.cat(confi_targets,0)
        new_confi_preds = torch.cat(confi_preds,0)

        positives = new_confi_targets > 0
        negatives = new_confi_targets == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()


        pos_normalizer = positives.sum().float()+1
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        weights = torch.ones_like(new_confi_preds)*cls_weights

        loss = self.loss_fn(new_confi_preds,new_confi_targets,weights).sum()

        return loss

    def sampling(self,batch_dict,training=False):

        anchors=[]
        confi_targets = []

        batch_cls = torch.sigmoid(batch_dict['batch_cls_preds'].squeeze(-1))

        for i in range(len(batch_cls)):
            boxe_preds = batch_dict['batch_box_preds'][i]
            confi = batch_cls[i]

            mask = confi>self.train_score

            boxe_preds=boxe_preds[mask]
            confi=confi[mask]

            if training:
                cur_gt = batch_dict['gt_boxes'][i]

                cnt = cur_gt.__len__() - 1
                while cnt > 0 and cur_gt[cnt].sum() == 0:
                    cnt -= 1
                cur_gt = cur_gt[:cnt + 1]
                boxe_preds = torch.cat([boxe_preds,cur_gt[:,0:7]])

                ious = batch_dict['gt_ious'][i]
                ious = ious[mask]

                ious[ious>=0.7] = 1
                ious[ious<0.7] = 0

                one = confi.new_ones(cur_gt.shape[0])
                cur_confi_targets = torch.cat([ious,one],0)
                confi_targets.append(cur_confi_targets)

            anchors.append(boxe_preds)


        return anchors, confi_targets

    def obtain_conf_preds(self, confi_im, anchors):

        confi = []

        for i, im in enumerate(confi_im):
            boxes = anchors[i]
            im = confi_im[i]
            if len(boxes) == 0:
                confi.append(torch.empty(0).type_as(im))
            else:
                (xs, ys) = self.gen_grid_fn(boxes[:, [0, 1, 3, 4, 6]])
                out = bilinear_interpolate_torch_gridsample(im, xs, ys)
                x = torch.mean(out, 0).view(-1)
                confi.append(x)

        return confi

    def forward(self, batch_dict):

        anchors, confi_targets = self.sampling(batch_dict, training=self.training)

        batch_im = self.convs_1(batch_dict['st_features_2d'])
        confi_im = self.convs_conf(batch_im)

        confi_preds = self.obtain_conf_preds(confi_im, anchors)

        if self.training:
            self.forward_dict["confi_targets"] = confi_targets
            self.forward_dict["confi_preds"] = confi_preds

        batch_size = len(confi_preds)
        max_len = 0#batch_dict['batch_box_preds'].shape[1]

        for i in range(batch_size):
            if len(confi_preds[i])>max_len:
                max_len = len(confi_preds[i])

        batch_boxes_preds = []
        batch_boxes_cls = []

        for i in range(batch_size):
            boxes = anchors[i]
            cls = confi_preds[i]
            cls = torch.sigmoid(cls)
            new_boxes = boxes.new_zeros(size=(max_len, boxes.shape[-1]))
            new_boxes[:boxes.shape[0], :] = boxes
            new_conf = cls.new_zeros(size=(max_len,))
            new_conf[:boxes.shape[0]] = cls
            batch_boxes_preds.append(new_boxes)
            batch_boxes_cls.append(new_conf)

        batch_boxes_preds = torch.stack(batch_boxes_preds, dim=0)
        batch_boxes_cls = torch.stack(batch_boxes_cls, dim=0)

        batch_dict['batch_cls_preds'] = batch_boxes_cls.unsqueeze(-1)
        batch_dict['batch_box_preds'] = batch_boxes_preds


        return batch_dict
