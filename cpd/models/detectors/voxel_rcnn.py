from .detector3d_template import Detector3DTemplate
import time
class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            #print(str(cur_module)[:10])

            #begin=time.time()
            batch_dict = cur_module(batch_dict)
            #end=time.time()
            #print(end-begin)

        if self.training:

            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts, = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, batch_dict

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)#

        loss =  loss_rpn + loss_rcnn#
        return loss, tb_dict, disp_dict

    def get_training_semi_loss(self):
        disp_dict = {}
        loss_rcnn, tb_dict= self.roi_head.get_semi_loss()#
        loss = loss_rcnn

        return loss, tb_dict, disp_dict
