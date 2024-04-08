from cpd.unsupervised_core.dbscan import DBSCAN
from cpd.unsupervised_core.oyster import OYSTER
from cpd.unsupervised_core.mfcf import MFCF
from cpd.unsupervised_core.c_proto_refine import C_PROTO
all_init = {
    'DBSCAN': DBSCAN,
    'OYSTER': OYSTER,
    'MFCF': MFCF,
}


all_refine = {
    'C_PROTO': C_PROTO,
}

def compute_outline_box(seq_name, root_path, dataset_cfg):
    suc = None
    if 'InitLabelGenerator' in dataset_cfg:
        print('run init outliner')
        init_method = dataset_cfg['InitLabelGenerator']
        outliner = all_init[init_method](seq_name, root_path, dataset_cfg)
        suc = outliner()
    if 'LabelRefiner' in dataset_cfg:
        print('run refiner')
        refine_method = dataset_cfg['LabelRefiner']
        refiner = all_refine[refine_method](seq_name, root_path, dataset_cfg)
        suc = refiner()
    return suc
