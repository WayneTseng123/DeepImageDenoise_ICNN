import time
import numpy as np

from utils import *

def inception_layer( conv_33_size,
                   conv_55_size, conv_77_size,conv_99_size
                    layer_dict, inputs=None,
                    bn=False, wd=0, init_w=None,
                    pretrained_dict=None, trainable=True, is_training=True,
                    name='inception'):
    
    if inputs is None:
        inputs = layer_dict['cur_input']
    layer_dict['cur_input'] = inputs

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], layer_dict=layer_dict, pretrained_dict=pretrained_dict,
                   bn=bn, nl=tf.nn.relu, init_w=init_w, trainable=trainable,
                   is_training=is_training, wd=wd, add_summary=False):


      

        L.conv(filter_size=1, out_dim=conv_33_reduce_size,
               inputs=inputs, name='{}_3x3_reduce'.format(name))
        conv_33 = L.conv(filter_size=3, out_dim=conv_33_size,
                         name='{}_3x3'.format(name))

        L.conv(filter_size=1, out_dim=conv_55_reduce_size,
               inputs=inputs, name='{}_5x5_reduce'.format(name))
        conv_55 = L.conv(filter_size=5, out_dim=conv_55_size,
                         name='{}_5x5'.format(name))
         L.conv(filter_size=1, out_dim=conv_77_reduce_size,
               inputs=inputs, name='{}_5x5_reduce'.format(name))
        conv_77 = L.conv(filter_size=5, out_dim=conv_55_size,
                         name='{}_5x5'.format(name))
         L.conv(filter_size=1, out_dim=conv_99_reduce_size,
               inputs=inputs, name='{}_5x5_reduce'.format(name))
        conv_99 = L.conv(filter_size=5, out_dim=conv_55_size,
                         name='{}_5x5'.format(name))
       

        output = tf.concat([conv_33, conv_55, conv_77, conv_99], 4,
                           name='{}_concat'.format(name))
        layer_dict['cur_input'] = output
        layer_dict[name] = output
    return output

def multicnn(layer_dict, inputs=None, is_training=True, output_channels=1):

    if inputs is None:
        inputs = layer_dict['cur_input']
    layer_dict['cur_input'] = inputs

    for layers in xrange(2, 7 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 128, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block8'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same')
    return input - output


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, sigma=25, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.sigma = sigma
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = self.Y_ + tf.random_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        # self.X = self.Y_ + tf.truncated_normal(shape=tf.shape(self.Y_), stddev=self.sigma / 255.0)  # noisy images
        self.Y = multicnn(self.X, is_training=self.is_training)
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.SGDOptimizer(self.lr, name='SGDOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, test_data, sample_dir, summary_merged, summary_writer):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in xrange(len(test_data)):
            clean_image = test_data[idx].astype(np.float32) / 255.0
            output_clean_image, noisy_image, psnr_summary = self.sess.run(
                [self.Y, self.X, summary_merged],
                feed_dict={self.Y_: clean_image,
                           self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)
            groundtruth = np.clip(test_data[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),
                        groundtruth, noisyimage, outputimage)
        avg_psnr = psnr_sum / len(test_data)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def denoise(self, data):
        output_clean_image, noisy_image, psnr = self.sess.run([self.Y, self.X, self.eva_psnr],
                                                              feed_dict={self.Y_: data, self.is_training: False})
        return output_clean_image, noisy_image, psnr

    def train(self, data, eval_data, batch_size, ckpt_dir, epoch, lr, sample_dir, eval_every_epoch=2):
        # assert data range is between 0 and 1
        numBatch = int(data.shape[0] / batch_size)
        # load pretrained model
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
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        writer = tf.summary.FileWriter('./logs', self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)  # eval_data value range is 0-255
        for epoch in xrange(start_epoch, epoch):
            np.random.shuffle(data)
            for batch_id in xrange(start_step, numBatch):
                batch_images = data[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                # batch_images = batch_images.astype(np.float32) / 255.0 # normalize the data to 0-1
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.lr: lr[epoch],
                                                            self.is_training: True})
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data, sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='multinet-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, test_files, ckpt_dir, save_dir):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.sigma) + " start testing...")
        for idx in xrange(len(test_files)):
            clean_image = load_images(test_files[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image = self.sess.run([self.Y, self.X],
                                                            feed_dict={self.Y_: clean_image, self.is_training: False})
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'noisy%d.png' % idx), noisyimage)
            save_images(os.path.join(save_dir, 'denoised%d.png' % idx), outputimage)
        avg_psnr = psnr_sum / len(test_files)
        print("--- Average PSNR %.2f ---" % avg_psnr)
