import numpy as np
import tensorflow.compat.v1 as tf
import tf_lib
import tensorflow_probability as tfp
from tqdm import tqdm
tfd = tfp.distributions

def kl_divergence(mu1, logv1, mu2, logv2):
    '''
    Returns KL Divergence (N(mu1, sigma1) || N(mu2, sigma2))
    '''
    return 0.5 * (tf.reduce_sum(logv2-logv1, axis=-1) - tf.cast(tf.shape(logv2)[1], tf.float32)\
        + tf.reduce_sum(tf.math.exp(logv1-logv2), axis=-1) + tf.reduce_sum((mu1-mu2)**2/tf.math.exp(logv2), axis=-1))


def one_hot(data):
    one_hot_targets = np.eye(10)[data]
    return one_hot_targets


class CVAE(tf_lib.trainer.Trainer):
    def __init__(self, sess, model, params, load_data_f, name=None, mode="train"):
        '''
        sess: tf.Session() object
        model: dictionary of models to be trained
        params: HParams object, parameters for training and model
        load_data_f: function that loads dataset
        name: name of the model (variable_scope under which model parameters will be saved)
        mode: train | eval
        '''
        self.model = model
        tf_lib.trainer.Trainer.__init__(self, sess, params, load_data_f, name, mode)

    def build_graph(self):
        self.condition = tf.placeholder(shape=(None, *self.params.condition_size), dtype=tf.float32, name='x')
        self.target = tf.placeholder(shape=(None, *self.params.target_size), dtype=tf.float32, name='y')
        self.is_training = tf.placeholder(shape=(), dtype=tf.bool, name='is_training')
        self.lr = tf.placeholder(shape=(), dtype=tf.float32, name='lr')

        self.mu_prior, self.logv_prior = self.model['prior_net'](self.condition)
        self.mu_posterior, self.logv_posterior = self.model['posterior_net']((self.condition, self.target))

        self.eps = tf.random.normal(shape=(tf.shape(self.mu_posterior)[0], self.params.latent_dim))

        self.latent_var = tf.math.exp(0.5 * self.logv_posterior) * self.eps + self.mu_posterior

        self.target_logits = tf.nn.sigmoid(self.model['generator_net']((self.condition, self.latent_var)))
        self.target_flattened = tf.reshape(self.target, (-1, np.prod(self.params.target_size)))

        self.kl = kl_divergence(self.mu_posterior, self.logv_posterior, self.mu_prior, self.logv_prior)
        self.log_prob = tf.reduce_sum(self.target_flattened * tf.math.log(self.target_logits + 1e-12) + (1. - self.target_flattened) * tf.math.log(1.-self.target_logits + 1e-12), axis=-1)

        self.loss = self.kl - self.log_prob

        self.loss = tf.reduce_mean(self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.step = self.optimizer.minimize(self.loss, var_list=tf.trainable_variables())

        self.init_op = tf.global_variables_initializer()

        self.add_summary()
        self.summary = tf.summary.merge_all()

    def add_summary(self):
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("KL", tf.reduce_mean(self.kl))
        tf.summary.scalar("NLL", -tf.reduce_mean(self.log_prob))

    def train_impl(self, start_epoch):
        for epoch in range(start_epoch, self.params.epochs):
            train_iterator = tf_lib.datasets.dataset_iterator(self.train_data, self.params.batch_size)
            with tqdm(total=self.iters) as pbar:
                for i, data in enumerate(train_iterator):
                    _, summ_, kl_, log_prob_, loss_, logits_, target_ = self.sess.run([
                        self.step, self.summary, self.kl, self.log_prob, self.loss, self.target_logits, self.target_flattened], feed_dict={
                        self.lr:self.params.lr,
                        self.is_training:True,
                        self.condition:one_hot(data[1]),
                        self.target:data[0]
                    })

                    pbar.update(1)
                    pbar.set_postfix(loss=loss_, kl=np.mean(kl_), nll=np.mean(-log_prob_))
                    self.writer.add_summary(summ_, self.counter)
                    self.writer.flush()
                    self.counter+=1
            print("[*] Epoch {}/{} completed".format(epoch + 1, self.params.epochs))
            self.eval()
            mu_posterior, logv_posterior = self.sess.run([self.mu_posterior, self.logv_posterior], feed_dict={self.target:data[0], self.condition:one_hot(data[1])})
            mu_prior, logv_prior = self.sess.run([self.mu_prior, self.logv_prior], feed_dict={self.condition:one_hot(data[1])})
            print("[*] Posterior Mu {}, Logv {}".format(np.mean(mu_posterior), np.mean(logv_posterior)))
            print("[*] Prior Mu {}, Logv {}".format(np.mean(mu_prior), np.mean(logv_prior)))

    def eval(self):
        test_iterator = tf_lib.datasets.dataset_iterator(self.test_data, self.params.batch_size)
        total_loss = 0.
        total_kl = 0.
        total_nll = 0.
        total = 0.
        for i, data in enumerate(test_iterator):
            summ_, kl_, log_prob_, loss_ = self.sess.run([
                self.summary, self.kl, self.log_prob, self.loss],
                feed_dict={
                    self.condition:one_hot(data[1]),
                    self.target: data[0]
                })
            total_loss += loss_ * len(data)
            total_kl += np.mean(kl_) * len(data)
            total_nll += np.mean(-log_prob_) * len(data)
            total += len(data)
            self.test_writer.add_summary(summ_, self.counter)
            self.test_writer.flush()
        print("[*] Evaluated Loss:{}, KL:{}, NLL: {}".format(total_loss/total, total_kl/total, total_nll/total))

    def generate_samples(self, condition):
        mu_prior, logv_prior = self.sess.run([self.mu_prior, self.logv_prior], feed_dict={self.condition:condition})
        epsilon = np.random.randn(*mu_prior.shape)
        latent_z = epsilon * np.exp(0.5 * logv_prior) +  mu_prior
        logits = self.sess.run(self.target_logits, feed_dict={self.condition:condition, self.latent_var:latent_z})
        logits = np.reshape(logits, (-1, 28, 28))
        return logits
