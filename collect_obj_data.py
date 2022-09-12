# import os, glob, joblib, numpy as np; from tqdm import tqdm
# data_dir = '/disk2/nuri/AI2Thor_offline_data_2.0.2'
# scenes = os.listdir(data_dir)
# data_list = []
# for scene in scenes:
#     data_list.extend(glob.glob(f"{data_dir}/{scene}/bboxes/*|0.dat.gz"))
#     os.makedirs(f"{data_dir}/{scene}/objects", exist_ok=True)
#
# for index in tqdm(range(len(data_list))):
#     bbox_data = joblib.load(data_list[index])
#     for bbox_i, bbox in enumerate(bbox_data['bboxes']):
#         bbox_data_i = {}
#         bbox_data_i['bboxes'] = bbox_data['bboxes'][bbox_i]
#         bbox_data_i['bbox_categories'] = bbox_data['bbox_categories'][bbox_i]
#         bbox_data_i['bbox_locations'] = bbox_data['bbox_locations'][bbox_i]
#         bbox_data_i['bbox_dists'] = bbox_data['bbox_dists'][bbox_i]
#         joblib.dump(bbox_data_i, data_list[index].replace("bboxes", "objects").replace("|0.dat.gz",f"|0|{str(bbox_i).zfill(3)}.dat.gz"))

import os, glob, joblib, numpy as np; from tqdm import tqdm;
data_dir = '/disk2/nuri/AI2Thor_offline_data_2.0.2'
scenes = os.listdir(data_dir)
data_list = []
for scene in scenes:
    data_list.extend(glob.glob(f"{data_dir}/{scene}/objects/*"))

for index in tqdm(range(len(data_list))):
    data_name = data_list[index]
    bbox_data = joblib.load(data_name)
    bbox_size = (bbox_data['bboxes'][2] - bbox_data['bboxes'][0]) * (bbox_data['bboxes'][3] - bbox_data['bboxes'][1])
    category = bbox_data['bbox_categories']
    location = bbox_data['bbox_locations']
    if data_name.count('|') == 4:
        os.rename(data_name, data_name.replace(".dat.gz",f"|{bbox_data['bbox_categories']}|{location[0]}|{location[1]}|{location[2]}|{str(bbox_data['bbox_dists']):.4}|{str(bbox_size):.5}.dat.gz"))
    elif data_name.count('|') == 8:
        os.rename(data_name, data_name.replace(".dat.gz",f"|{str(bbox_data['bbox_dists']):.4}|{str(bbox_size):.5}.dat.gz"))
