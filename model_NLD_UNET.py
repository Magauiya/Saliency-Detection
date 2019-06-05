import time

from utils import *
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
import time

class NLD_UNET_SD(object):
    def __init__(self, sess, input_c_dim=3, batch_size=1, IMG_SIZE=(256, 256)):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.IMG_SIZE = IMG_SIZE
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        
        # Placeholders
        self.input  = tf.placeholder(tf.float32, [None, IMG_SIZE[0], IMG_SIZE[1], self.input_c_dim], name='input_img')
        self.logit = NLD_UNET(self.input, is_training=self.is_training, output_channels = 1)                            # logit
        self.saliency_map = tf.nn.sigmoid(self.logit)                                                                   # predicted saliency map
        self.label = tf.placeholder(tf.float32, [None, IMG_SIZE[0], IMG_SIZE[1], 1], name='gt_map')                     # label saliency map
        
        # Loss function = cross entropy, Eval metrics = MAE
        self.loss = (1./batch_size) * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.label, logits=self.logit, name='cross_entropy_loss'))
        self.eva_mae = tf.reduce_mean(tf.abs(self.label - self.saliency_map), name = 'MAE')
        
        # Cost Minimizer
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name = 'AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
                    
        # Initializing global variables
        self.saver = tf.train.Saver(max_to_keep=10)
        init_global = tf.global_variables_initializer()
        self.sess.run(init_global)
        print("[*] Initialize model successfully...")
    
    ########## TRAINING STAGE ##########
    def train(self, data_dir, eval_data_dir, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=1):
        # Loading trainlist
        train_list = get_img_list(data_dir)
        numBatch = int(10000 / batch_size) # MSRA 10k images
        
        # Loading trained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Could not find pretrained model!")
        
        # Summary for analysis - Loss, MAE, learning rate
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        summary_mae = tf.summary.scalar('MAE', self.eva_mae)
        summary_loss = tf.summary.scalar('Loss', self.loss)
        summary_lr = tf.summary.scalar('Learning rate', self.lr)
        merged_train = tf.summary.merge([summary_loss, summary_lr], name='Statistics for Training stage')

        # Performance Evaluation using Validation set
        self.evaluate(iter_num, eval_data_dir, sample_dir=sample_dir, merged_test=summary_mae, summary_writer=writer)
        
        # Network training stage
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        for epoch in xrange(start_epoch, epoch):
            np.random.shuffle(train_list)
            for batch_id in xrange(start_step, numBatch):
                (input_batch, mask_batch) = get_image_batch(train_list, batch_id*batch_size, batch_size=batch_size)
                # Normalization
                mask_batch = mask_batch.astype(np.float32)
                mask_batch = mask_batch / 255.
                input_batch = input_batch.astype(np.float32)
                input_batch = input_batch / 255.
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged_train], feed_dict={self.label: mask_batch,
                                                                                                      self.input: input_batch,
                                                                                                      self.lr: lr[epoch],
                                                                                                      self.is_training: True})
                if np.mod(batch_id+1, 50) == 0:
                    print("Epoch: [%2d] [%4d/%4d] Time: %4.4f, Loss: %.6f" % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))

                iter_num += 1
                writer.add_summary(summary, iter_num)
            
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data_dir, sample_dir=sample_dir,  merged_test=summary_mae, summary_writer=writer)
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training. U-NET")
            
    def evaluate(self, iter_num, valid_path, sample_dir, merged_test, summary_writer):
        valid_list = get_img_list(valid_path)
        tf_mae_sum = 0
        
        print("---------------------- Validation ----------------------")
        for idx in xrange(len(valid_list)):
            # Loading Validation set
            (input_batch, mask_batch) = get_image_batch(valid_list, idx, batch_size = 1)
            mask_batch = mask_batch.astype(np.float32)
            mask_batch = mask_batch / 255.
            input_batch = input_batch.astype(np.float32)
            input_batch = input_batch / 255.

            output_smap, input_img, tf_mae, mae_summary = self.sess.run([self.saliency_map, self.input, self.eva_mae, merged_test], feed_dict={self.label: mask_batch,
                                                                                                                                                 self.input: input_batch,
                                                                                                                                                 self.is_training: False})
            tf_mae_sum += tf_mae.astype(np.float32)
            summary_writer.add_summary(mae_summary, iter_num)
    
            print("ID: %d saliency map range: %.3f-%.3f mae: %.3f" % (idx+1, np.amin(output_smap), np.amax(output_smap), tf_mae))

            if np.mod(idx + 1, 5) == 0:
                save_images(os.path.join(sample_dir, 'test%d_%d.jpg' % (idx + 1, iter_num)), mask_batch, input_img, output_smap)
        
        print("AVERAGE MAE: %.3f " % (tf_mae_sum / len(valid_list)))
        print("----------------------*---------*----------------------")

    ########## TESTING STAGE ##########
    def inference(self, test_path, ckpt_dir, save_test_dir):
        # Loading ckpt from ckpr_dir and initializing the network
        tf.global_variables_initializer().run()
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Error! CKPT Loading status - FAILED'
        print('[*] CKPT Loading status - SUCCESS')
        
        # Performance metrics
        mae_sum = 0
        recall_sum = 0
        precision_sum = 0
        avg_cnn_time = 0
        avg_all_time = 0
        
        # loading list of test images
        test_list = get_test_list(test_path)
        print 'Testset: ', len(test_list), ' imgs'
        print("---------------------- INFERENCE ----------------------")
        for idx in xrange(len(test_list)):
            
            # ground truth map and input image
            (test_image, test_map) = get_test_image(test_list, idx)
            
            # Resizing input test image, normalizing to feed to the network
            start_all = time.time()
            map_cnn   = np.array(test_map.resize((256, 256),   Image.BICUBIC)).astype(np.float32) / 255.
            input_cnn = np.array(test_image.resize((256, 256), Image.BICUBIC)).astype(np.float32) / 255.
            map_cnn = np.expand_dims(np.expand_dims(map_cnn, axis=0), axis=3)
            input_cnn = np.expand_dims(input_cnn, axis=0)
            
            # Predicted Saliency Map
            start_cnn = time.time()  #
            output_cnn= self.sess.run(self.saliency_map, feed_dict={self.label: map_cnn, self.input: input_cnn, self.is_training: False})
            end_cnn = time.time()
            
            output_cnn = np.clip(np.squeeze(output_cnn)*255., 0, 255).astype(np.uint8)
            saliency_map =Image.fromarray(output_cnn).resize(test_image.size, Image.BICUBIC)
            end_all = time.time()
            
            avg_cnn_time = avg_cnn_time + (end_cnn - start_cnn)
            avg_all_time = avg_all_time + (end_all - start_all)
 
            # Performance analysis
            test_map_np = np.array(test_map).astype(np.float32) / 255.
            saliency_map_np = np.array(saliency_map).astype(np.float32) / 255.
            mae = np.mean(np.abs(test_map_np - saliency_map_np))
            mae_sum += mae
            recall, precision = prec_recall_estimate(test_map_np, saliency_map_np)
            precision_sum += precision
            recall_sum += recall
            print("ID: %d saliency map range: %.3f-%.3f mae: %.3f recall: %.3f precision: %.3f" % (idx+1, np.amin(saliency_map), np.amax(saliency_map), mae, recall, precision))

            # Saving results
            if idx <= 1000 and idx > 100:
                saliency_map.save(os.path.join(save_test_dir, '%d_predicted.png' % (idx + 1)))
                test_map.save(os.path.join(save_test_dir, '%d_ground_truth.png' % (idx + 1)))
                test_image.save(os.path.join(save_test_dir, '%d_rgb_image.png' % (idx + 1)))
    
        f1score = f1score_estimate(precision_sum/len(test_list), recall_sum/len(test_list))
        print("AVERAGE MAE: %.5f F1 Score: %.5f" % (mae_sum/len(test_list), f1score))
        print("AVERAGE Time (network): %.5f sec/img Time(resize->network->resize): %.5f sec/img" % (avg_cnn_time/len(test_list), avg_all_time/len(test_list)))
        print("----------------------*---------*----------------------")

        
    def save(self, iter_num, ckpt_dir, model_name='NLD-UNET_SD-TF'):
        saver = tf.train.Saver(max_to_keep=10)
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        self.saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            print global_step
            self.saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

