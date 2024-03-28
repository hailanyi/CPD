from pcdet.unsupervised_core.dbscan import DBSCAN
from pcdet.unsupervised_core.ppscore_fit import PPSCORE
from pcdet.unsupervised_core.oyster import OYSTER
from pcdet.unsupervised_core.mfcf import MFCF
from pcdet.unsupervised_core.c_proto_refine import C_PROTO
from pcdet.unsupervised_core.c_proto_self_improve import C_PROTO_SI
from pcdet.unsupervised_core.pseudo_label import PseudoLabel
from pcdet.unsupervised_core.pseudo_label_s import PseudoLabelS
all_init = {
    'DBSCAN': DBSCAN,
    'PPSCORE': PPSCORE,
    'OYSTER': OYSTER,
    'MFCF': MFCF,
    'PseudoLabel': PseudoLabel,
    'PseudoLabelS': PseudoLabelS
}


all_refine = {
    'C_PROTO': C_PROTO,
    'C_PROTO_SI': C_PROTO_SI
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
