__author__ = 'Magauiya Zhussip. UNIST, South Korea. All rights reserved 2018'

import argparse
from glob import glob
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from model_NLD_UNET import NLD_UNET_SD
from utils import *

parser = argparse.ArgumentParser(description='')

# TRAINING PARAMETERS
parser.add_argument('--epoch', dest='epoch', type=int, default=2, help='# of epoch')
parser.add_argument('--batch', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for optimizer')
parser.add_argument('--gpu', dest='gpu', type=int, default=1, help='which gpu to use')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')

# SAVE Results and CKPT
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./predictions/training/', help='validation samples are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./predictions/testing/', help='test samples are saved here')

# PATH for Training, Validation, Testing sets
parser.add_argument('--train_path', dest='train_set', default='/MSRA10k/trainset', help='dataset for training')
parser.add_argument('--valid_path', dest='eval_set', default='/MSRA10k/validset', help='dataset for validation')
parser.add_argument('--test_path', dest='test_set', default='/ECSSD', help='dataset for testing: DUTOMRON or ECSSD')
args = parser.parse_args()

def main(_):
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
	if not os.path.exists(args.test_dir):
		os.makedirs(args.test_dir)

    # Learning rate
    if args.phase == 'train':
        lr = args.lr * np.ones([args.epoch])
        lr[20:] = lr[0] / 10.0
        lr[40:] = lr[21] / 10.0
        lr[60:] = lr[41] / 10.0
    
    # Output parameters used for training/testing
    parameters(phase = args.phase, batch_size=args.batch_size, learning_rate=args.lr, epochs=args.epoch, input_ch=3, train_path=args.train_set, valid_path=args.eval_set, test_path=args.test_set)
    
    # Testing/Training model using GPU
    print('GPU %d is ACTIVATED!' % args.gpu)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        model = NLD_UNET_SD(sess)
        if args.phase == 'train':
            NLD_UNET_SD_train(model, lr=lr)
        elif args.phase == 'test':
            NLD_UNET_SD_test(model)
        else:
            print('[!]Unknown phase')
            exit(0)

def NLD_UNET_SD_train(NLD_UNET_SD, lr):
    valid_path = args.eval_set
    train_path = args.train_set
    NLD_UNET_SD.train(train_path, valid_path, batch_size=args.batch_size, ckpt_dir=args.ckpt_dir, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir)

def NLD_UNET_SD_test(NLD_UNET_SD):
    NLD_UNET_SD.inference(test_path = args.test_set, ckpt_dir=args.ckpt_dir, save_test_dir=args.test_dir)

def parameters(phase, batch_size, learning_rate, epochs, input_ch, train_path, valid_path, test_path):
    print("****************** PARAMETERS ******************")
    print 'Phase: ', phase
    print 'Batch size: ', batch_size
    print 'Init. learning rate: ', learning_rate
    print 'Optimizer: ', 'Adam'
    print 'Epochs: ', epochs
    print 'Input img channel: ', input_ch
    print 'Trainset path: ', train_path
    print 'Validation set path: ', valid_path
    print 'Testset path: ', test_path
    print("******************-********-******************")

if __name__ == '__main__':
    tf.app.run()
