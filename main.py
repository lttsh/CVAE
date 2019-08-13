import numpy as np
import tensorflow as tf

import tf_lib
from CVAE import cvae_trainer
from CVAE import models
from tf_lib.utils import visualize_images
import tf_lib

def preproc(x):
    return x > 127

hparams = tf.contrib.training.HParams(
    num_val=10000,
    num_labels=None,
    batch_size=64,
    latent_dim=200,
    debug=True,
    log_dir='logs',
    experiment_name='quadrant_test',
    condition_size=(28, 28),
    target_size=(28, 28),
    num_hidden=1000,
    epochs=20,
    lr=1e-2,
    log_freq=20,
    preproc=preproc,
)

# tf_lib.utils.load_params(hparams, args, None)
train_data, val_data, test_data = tf_lib.loaders.load_mnist(**hparams.values())
tf_lib.utils.visualize_images(train_data[0][:64] * 255, 'mnist.jpg')
tf_lib.utils.visualize_images(cvae_trainer.quadrant_tl(train_data[0][:64]) * 255, 'quadrant.jpg')

model = {
    'prior_net': models.PriorNetwork(hparams),
    'posterior_net': models.PosteriorNetwork(hparams),
    'generator_net': models.GenerationNetwork(hparams),
}
with tf.Session() as sess:
    trainer = cvae_trainer.CVAE(sess, model, hparams, tf_lib.loaders.load_mnist)
    tf_lib.utils.show_all_variables()
    trainer.train()

    test_iterator = tf_lib.datasets.dataset_iterator(trainer.test_data, trainer.params.batch_size)
    batch = next(test_iterator)[0]
    print(batch.size)
    conditions = cvae_trainer.quadrant_tl(batch)
    logits = trainer.generate_samples(conditions) * 255
    visualize_images(logits, 'results_quadrant.jpg', num_rows=8)
