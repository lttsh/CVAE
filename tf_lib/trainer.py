import numpy as np
import os
import tensorflow as tf
from tf_lib.utils import show_variables, safe_dir

class Trainer(object):
    '''
    Generic trainer class handling logs and checkpoints and data loading (CIFAR100)
    '''
    def __init__(self, sess, params, load_data_f, name=None, mode="train"):
        '''
        Init:
        sess: tf.Session()
        params: HParams object containing hyperparameters the model and training.
        name: scope of variables to save, if None save all.
        mode: "train" or "eval"
        '''
        self.sess = sess
        self.params = params
        self.name = name
        self.debug = params.debug
        self.batch_size = params.batch_size
        self.checkpoint_dir = safe_dir(self.params.log_dir, self.params.experiment_name)

        if load_data_f:
            self.train_data, self.val_data, self.test_data = load_data_f(**params.values())
            self.iters = len(self.train_data[0]) // self.batch_size # Total number of iterations to go through one batch
            if self.iters * self.batch_size < len(self.train_data[0]):
                self.iters += 1

        self.build_graph()
        self.mode = mode
        self.test_writer = None

    def save_params(self):
        import json
        import os
        pass
        # with open(self.checkpoint_dir + '/params.json', 'w') as fp:
        #     fp.write(self.params.to_json(indent=4, sort_keys=True))
        # print("[*] Parameters saved in {}".format(self.checkpoint_dir + '/params.json'))

    def build_graph(self):
        '''
        Builds computation graph for training and evaluating
        '''
        return NotImplementedError

    def train_impl(self, start_epoch):
        raise NotImplementedError

    def train(self):
            try:
                self.writer = tf.summary.FileWriter(self.checkpoint_dir + '/train', self.sess.graph)
                self.test_writer = tf.summary.FileWriter(self.checkpoint_dir + '/test', self.sess.graph)
                self.sess.run(self.init_op)
                # restore check-point if it exists
                start_epoch = self.load_checkpoint()
                self.train_impl(start_epoch)
            except KeyboardInterrupt:
                print("Interupted... saving model.")
            self.save()

    def eval(self, partial=None, use_val=True, write=True):
        raise NotImplementedError

    def save(self):
        experiment_name = self.params.experiment_name.split('/')[0]
        print("Experiment name", experiment_name)

        self.saver.save(
            self.sess,
            os.path.join(self.checkpoint_dir, experiment_name + '.model'),
            global_step=self.counter)
        print("[*] Saved model in {}".format(self.checkpoint_dir))

    def optimistic_restore(self, save_file):
        # https://gist.github.com/iganichev/d2d8a0b1abc6b15d4a07de83171163d4
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for
                          var in tf.global_variables()
                          if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0],
                              tf.global_variables()),
                          tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
              curr_var = name2var[saved_var_name]
              var_shape = curr_var.get_shape().as_list()
              if var_shape==saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, save_file)

    def load(self):
        import re
        print(" [*] Reading checkpoints from {}".format(self.checkpoint_dir))

        if self.name is not None:
            print("Name is", self.name)
            variables_to_load = [v for v in tf.global_variables() if self.name in v.name]
            if self.debug:
                print("Loading and saving variables ... ")
                show_variables(variables_to_load)
            self.saver = tf.train.Saver(variables_to_load)
        else:
            if self.debug:
                print("Loading and saving variables ... ")
                show_variables(tf.global_variables())
            self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.optimistic_restore(os.path.join(self.checkpoint_dir, ckpt_name))
            self.counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            print(self.counter)
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            # When no checkpoint is found, save parameters in the folder
            self.save_params()
            self.counter = 0
            return False

    def load_checkpoint(self):
        could_load = self.load()
        if could_load and self.mode !='eval':
            start_epoch = (int)(self.counter / self.iters)
        else:
            start_epoch = 0
        if could_load:
            print(" [*] Load SUCCESS")
        return start_epoch
