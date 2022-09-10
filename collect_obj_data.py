import os, glob, joblib, numpy as np; from tqdm import tqdm
data_dir = '/disk2/nuri/AI2Thor_offline_data_2.0.2'
scenes = os.listdir(data_dir)
data_list = []
for scene in scenes:
    data_list.extend(glob.glob(f"{data_dir}/{scene}/bboxes/*|0.dat.gz"))
    os.makedirs(f"{data_dir}/{scene}/objects", exist_ok=True)

for index in tqdm(range(len(data_list))):
    bbox_data = joblib.load(data_list[index])
    for bbox_i, bbox in enumerate(bbox_data['bboxes']):
        bbox_data_i = {}
        bbox_data_i['bboxes'] = bbox_data['bboxes'][bbox_i]
        bbox_data_i['bbox_categories'] = bbox_data['bbox_categories'][bbox_i]
        bbox_data_i['bbox_locations'] = bbox_data['bbox_locations'][bbox_i]
        bbox_data_i['bbox_dists'] = bbox_data['bbox_dists'][bbox_i]
        joblib.dump(bbox_data_i, data_list[index].replace("bboxes", "objects").replace("|0.dat.gz",f"|0|{str(bbox_i).zfill(3)}.dat.gz"))
