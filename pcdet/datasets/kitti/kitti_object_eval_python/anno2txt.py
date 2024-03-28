import pickle
import os


annos = pickle.load(open('result_merged.pkl','rb'))

for single_pred_dict in annos:
    frame_id = single_pred_dict['frame_id']
    out_path = 'final_data'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    cur_det_file = os.path.join( out_path,'%s.txt' % frame_id)
    with open(cur_det_file, 'w') as f:
        bbox = single_pred_dict['bbox']
        loc = single_pred_dict['location']
        dims = single_pred_dict['dimensions']  # lhw -> hwl

        for idx in range(len(bbox)):
            print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'#
                  % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                     bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                     dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                     loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx], single_pred_dict['score'][idx]),file=f)#