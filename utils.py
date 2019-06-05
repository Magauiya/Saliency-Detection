import gc
import os
import sys
import glob
import re
import math
import scipy.io
import imageio
import numpy as np
import tensorflow as tf
from PIL import Image

# *************** FOR TRAINING ***************
def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, '*'))
    l = [f for f in l if re.search("^\d+.png$", os.path.basename(f))]
    img_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + '_2.png'):
                img_list.append([f, f[:-4] + '_2.png', 2])
    return img_list

def get_image_batch(img_list, offset, batch_size):
    target_list = img_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    for pair in target_list:
        input_img = imageio.imread(pair[0])
        gt_img = imageio.imread(pair[1])
        input_list.append(input_img)
        gt_list.append(gt_img)

    input_list = np.array(input_list)
    input_list.resize([batch_size, 256, 256, 3])
    gt_list = np.array(gt_list)
    gt_list.resize([batch_size, 256, 256, 1])

    return (input_list, gt_list)

# *************** FOR TESTING ***************
def get_test_list(data_path):
    l = glob.glob(os.path.join(data_path, '*'))
    l = [f for f in l if re.search(".jpg$", os.path.basename(f))]
    img_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + '.png'):
                img_list.append([f, f[:-4] + '.png', 2])
    return img_list

def get_test_image(img_list, idx):
    target_list = img_list[idx]
    print target_list
    input_img = Image.open(target_list[0]).convert("RGB")
    gt_map = Image.open(target_list[1])
    return (input_img, gt_map)

def f1score_estimate(precision, recall):
    beta = 0.3
    f1score = ((1+beta)*precision*recall) / (beta*precision + recall)
    return f1score

def prec_recall_estimate(test_map, saliency_map):
    threshold = 2. * np.mean(saliency_map)
    beta = 0.3
    # making binary map
    saliency_map = (saliency_map > threshold).astype(np.int)
    test_map = test_map.astype(np.int)
    # vectorizing
    saliency_map = saliency_map.flatten()
    test_map = test_map.flatten()

    # calculating true positive, false negative, and false positive
    TP = np.sum(np.logical_and(saliency_map == 1, test_map == 1)).astype(np.float32)
    FP = np.sum(np.logical_and(saliency_map == 1, test_map == 0)).astype(np.float32)
    FN = np.sum(np.logical_and(saliency_map == 0, test_map == 1)).astype(np.float32)

    # calculating precision and recall
    if TP == 0 and (FP != 0 or FN != 0):
        precision = 0.               #Why? Because: https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        recall = 0.
    elif TP == 0 and FP == 0 and FN == 0:
        precision = 1.               #Why? Because: https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        recall = 1.
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

    return (recall, precision)

# *************** GENERAL ***************
def to_rgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    img[np.isnan(img)] = 0
    img = np.clip(img, 0, 1) # clipping between [0-1]
    img *= 255
    return img

def save_images(filepath, gt, data, pred):
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(data.reshape(-1, ny, ch)),
                          to_rgb(gt.reshape(-1, ny, 1)),
                          to_rgb(pred.reshape(-1, ny, 1))), axis=1)
                          
    Image.fromarray(img.round().astype(np.uint8)).save(filepath, 'JPEG', dpi=[250,250], quality=100)

