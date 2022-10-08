import os, glob, joblib;import numpy as np
from tqdm import tqdm
from scipy.stats import ks_2samp
import random
import cv2
import matplotlib.pyplot as plt
# mp3d_objectwise
data_dir = "/disk3/nuri/mp3d_with_obj_pcl_data_v2"
train_data_list = []
data_dir = os.path.join(data_dir, "train")
scenes = os.listdir(data_dir)
for scene in scenes:
    places = glob.glob(os.path.join(data_dir, scene) + "/*")
    for place in places:
        train_data_list.extend(glob.glob(place + "/*_rgb.png"))

bbox_categories = {}
bbox_hist = {}
for train_data in tqdm(train_data_list):
    x_obj = joblib.load(train_data.replace('_rgb.png', '.dat.gz'))['bbox_categories']
    hist, _ = np.histogram(x_obj, bins=40, range=(0,40))
    hist += 1
    hist = hist / np.sum(hist)
    bbox_categories[train_data] = x_obj
    bbox_hist[train_data] = hist
hists = np.stack(bbox_hist.values())
hists = np.array(hists, dtype=np.float16)
# dist = np.sum(hists[:,None] * np.log(hists[:, None]/(hists+0.0001)),-1)
# sim = -dist
# sim[np.eye(len(sim))==1] = 0

data = {}
data['hists'] = hists
data['data_path'] = train_data_list
joblib.dump(data, os.path.join(data_dir, "object_hists.dat.gz"))