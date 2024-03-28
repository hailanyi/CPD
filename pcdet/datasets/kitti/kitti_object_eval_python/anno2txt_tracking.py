import pickle
import os
from pathlib import Path


annos = pickle.load(open('result_merged_tracking.pkl','rb'))

for single_pred_dict in annos:
    frame_id = single_pred_dict['frame_id']
    seq_id = single_pred_dict['seq_id']
    out_path = Path('final_data_tracking')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if out_path is not None:
        os.makedirs(out_path / str(seq_id).zfill(4), exist_ok=True)
        cur_det_file = out_path / str(seq_id).zfill(4) / (str(frame_id).zfill(6) + '.txt')

        with open(cur_det_file, 'w') as f:
            bbox = single_pred_dict['bbox']
            loc = single_pred_dict['location']
            dims = single_pred_dict['dimensions']  # lhw -> hwl

            for idx in range(len(bbox)):
                print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                      % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                         bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                         dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                         loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                         single_pred_dict['score'][idx]), file=f)