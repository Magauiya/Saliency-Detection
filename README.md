# Saliency-Detection
Computer Vision Final project

The network is trained on MSRA10k dataset resized to [256, 256]. However, for testing phase, any image size is fine. 

Dependencies:
Tensorflow 1.5.0
Python 2.7 (3.0 should be ok)


To test the U-NLCF network, please write this command to the terminal:

python2 main.py --gpu=1 --phase='test' —test_path=“your test set path”

Please make sure that input image and its ground truth are in the same folder and input image should have ‘.jpg’ format, while its saliency map should be ‘.png’ format. Otherwise, we need to change reading function in utils.py. Output saliency map is stored automatically in predictions/testing/ folder.

To train:

python2 main.py --gpu=1 --phase=‘train’ —train_path=“your test set path” —valid_path=“your validation set path” 


The code consists of three parts:

main.py 			 - to set learning parameters and path
model_NLD_UNET.py - contains U-NLCS model and train/inference functions
utils.py			 - contains reading, storing, evaluating functions 

Thank you! 
Sincerely,
Magauiya Zhussip
