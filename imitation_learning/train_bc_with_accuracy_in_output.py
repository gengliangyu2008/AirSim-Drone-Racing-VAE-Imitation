import tensorflow as tf
import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))

# imports
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import racing_models
import racing_utils
from datetime import datetime

###########################################

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# DEFINE TRAINING META PARAMETERS
# base_dir = 'C:/tools/Drone_Racing_Files_v1/'
base_dir = 'C:/tools/Drone_Racing_Files_v1/'
il_data_dir = base_dir + 'il_datasets/'
data_dir_list = [il_data_dir + 'bc_v5_n0',
                 il_data_dir + 'bc_v5_n1',
                 il_data_dir + 'bc_v5_n2',
                 il_data_dir + 'bc_v5_n3']

training_mode = 'latent'  # 'full' or 'latent' or 'reg'

output_dir = base_dir + 'zz_model_outputs/bc_on_cmvae_con_50k/'
cmvae_weights_path = base_dir + 'zz_model_outputs/cmvae_con_50k_deeperResNet/cmvae_model_49.ckpt'

# cmvae_weights_path = base_dir + 'model_outputs/cmvae_unc/cmvae_model_65.ckpt'
# cmvae_weights_path = base_dir + 'model_outputs/cmvae_img/cmvae_model_45.ckpt'

# training_mode = 'reg'  # 'full' or 'latent' or 'reg'
reg_weights_path = base_dir + 'model_outputs/reg/reg_model_25.ckpt'

# training_mode = 'full'  # 'full' or 'latent' or 'reg'
# no auxiliary feature extraction weights

n_z = 10
batch_size = 32
epochs = 401
img_res = 64
max_size = None  # default is None
learning_rate = 1e-2  # 1e-2 for latent, 1e-3 for full

###########################################
# CUSTOM FUNCTIONS

@tf.function
def reset_metrics():
    train_loss_rec_v.reset_states()
    test_loss_rec_v.reset_states()
    train_accuracy_rec_v.reset_states()
    test_accuracy_rec_v.reset_states()


@tf.function
def compute_loss(labels, predictions):
    recon_loss = tf.losses.mean_squared_error(labels, predictions)
    return recon_loss

@tf.function
def compute_accuracy(labels, predictions):
    #recon_accuracy = tf.metrics.sparse_categorical_accuracy(labels, predictions)
    recon_accuracy = tf.metrics.categorical_accuracy(labels, predictions)
    return recon_accuracy


@tf.function
def train(images, labels, epoch, training_mode):
    with tf.GradientTape() as tape:
        if training_mode == 'full':
            predictions = bc_model(images)
        elif training_mode == 'latent':
            z, _, _ = cmvae_model.encode(images)
            predictions = bc_model(z)
        elif training_mode == 'reg':
            z = reg_model(images)
            predictions = bc_model(z)
        recon_loss = tf.reduce_mean(compute_loss(labels, predictions))
        recon_accuracy = tf.reduce_mean(compute_accuracy(labels, predictions))
    gradients = tape.gradient(recon_loss, bc_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, bc_model.trainable_variables))
    train_loss_rec_v(recon_loss)
    train_accuracy_rec_v(recon_accuracy)


@tf.function
def test(images, labels, training_mode):
    if training_mode == 'full':
        predictions = bc_model(images)
    elif training_mode == 'latent':
        z, _, _ = cmvae_model.encode(images)
        predictions = bc_model(z)
    elif training_mode == 'reg':
        z = reg_model(images)
        predictions = bc_model(z)
    recon_loss = tf.reduce_mean(compute_loss(labels, predictions))
    recon_accuracy = tf.reduce_mean(compute_accuracy(labels, predictions))
    test_loss_rec_v(recon_loss)
    test_accuracy_rec_v(recon_accuracy)

###########################################


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# allow growth is possible using an env var in tf2.0
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# load dataset
print('Starting dataset')
# train_ds, test_ds = racing_utils.dataset_utils.create_dataset_txt(data_dir, batch_size, img_res, data_mode='train')
train_ds, test_ds = racing_utils.dataset_utils.create_dataset_multiple_sources(data_dir_list, batch_size, img_res, data_mode='train', base_path=base_dir)
print('Done with dataset')

# create models
if training_mode == 'full':
    bc_model = racing_models.bc_full.BcFull()
elif training_mode == 'latent':
    # cmvae_model = racing_models.cmvae.Cmvae(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True)
    cmvae_model = racing_models.cmvae.CmvaeDirect(n_z=n_z, gate_dim=4, res=img_res, trainable_model=True)
    cmvae_model.load_weights(cmvae_weights_path)
    cmvae_model.trainable = False
    bc_model = racing_models.bc_latent.BcLatent()
elif training_mode == 'reg':
    reg_model = model = racing_models.dronet.Dronet(num_outputs=4, include_top=True)
    reg_model.load_weights(reg_weights_path)
    reg_model.trainable = False
    bc_model = racing_models.bc_latent.BcLatent()
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

# define metrics
train_loss_rec_v = tf.keras.metrics.Mean(name='train_loss_rec_v')
test_loss_rec_v = tf.keras.metrics.Mean(name='test_loss_rec_v')

train_accuracy_rec_v = tf.keras.metrics.Mean(name='train_accuracy_rec_v')
test_accuracy_rec_v = tf.keras.metrics.Mean(name='test_accuracy_rec_v')

metrics_writer = tf.summary.create_file_writer(output_dir)

# check if output folder exists
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# train
print('Start training ...')
flag = True
for epoch in range(epochs):
    # print('MODE NOW: {}'.format(mode))
    for train_images, train_labels in train_ds:
        train(train_images, train_labels, epoch, training_mode)
        if flag:
            bc_model.network.summary()
            flag = False
    for test_images, test_labels in test_ds:
        test(test_images, test_labels, training_mode)

    with metrics_writer.as_default():
        tf.summary.scalar('train_loss_rec_gate', train_loss_rec_v.result(), step=epoch)
        tf.summary.scalar('test_loss_rec_gate', test_loss_rec_v.result(), step=epoch)
        tf.summary.scalar('train_accuracy_rec_gate', train_accuracy_rec_v.result(), step=epoch)
        tf.summary.scalar('test_accuracy_rec_gate', test_accuracy_rec_v.result(), step=epoch)

    print('{} Epoch {} | Train L_gate: {} | Test L_gate: {} | Train A_gate: {} | Test A_gate: {}'
          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch,
                  train_loss_rec_v.result(), test_loss_rec_v.result(),
                  train_accuracy_rec_v.result(), test_accuracy_rec_v.result()
                  )
          )

    # save model
    if epoch > 0 and (epoch % 10 == 0 or epoch == epochs - 1):
        print('Saving weights to {}'.format(output_dir))
        bc_model.save_weights(os.path.join(output_dir, "bc_model_{}.ckpt".format(epoch)))

    reset_metrics()  # reset all the accumulators of metrics

print('bla')