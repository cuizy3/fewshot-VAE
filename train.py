import numpy as np
import matplotlib.pyplot as plt
import time

from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.datasets import mnist
import csv
import cv2
import glob
from imageio import imread, imwrite
import tensorflow as tf

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# 4C convolutions
num_classes = 200

# input image dimensions
img_rows, img_cols, img_chns = 84, 84, 3
#img_rows, img_cols, img_chns = 28, 28, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 16
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 256
intermediate_dim = 1024
epsilon_std = 1.0
epochs = 100

x = Input(shape=original_img_size)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden1 = Dense(intermediate_dim, activation='relu')(flat)
hidden2 = Dense(intermediate_dim, activation='relu')(hidden1)

z_mean = Dense(latent_dim)(hidden2)
z_log_var = Dense(latent_dim)(hidden2)


x_c = Input(shape=num_classes) # one-hot encoding
hiddenc1 = Dense(512, activation='relu')(x_c)
hiddenc2 = Dense(512, activation='relu')(hiddenc1)
hiddenc3 = Dense(512, activation='relu')(hiddenc2)
zprior_mean = Dense(latent_dim)(hiddenc3)
zprior_log_var = Dense(latent_dim)(hiddenc3)
# Use the "reparameterization trick" here to ensure that we can back-propagate the error through the sampling operation.  
# Notice that we never have to backpropagate "through" the sampling operation `epsilon`, only through `z_mean` and `z_log_var`.
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(None, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(None, latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * 42 * 42, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (None, filters, 42, 42)
else:
    output_shape = (None, 42, 42, filters)

decoder_reshape = Reshape(output_shape[1:])

#decoder_out = Conv2DTranspose(name='x_decoded', filters=3, kernel_size=1, strides=1, activation='sigmoid')(dec_out)
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters, num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (None, filters, 85, 85)
else:
    output_shape = (None, 85, 85, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')
# this should ressurect the original image size?

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)


def vae_loss(x, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    x = K.flatten(x)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * img_chns * metrics.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #zprior_mean, zprior_log_var
    #kl_loss = 0.5 * K.sum(-1 + zprior_log_var - z_log_var + K.exp(z_log_var) / K.exp(zprior_log_var) + K.square(z_mean - zprior_mean) / K.exp(zprior_log_var), axis=-1)
    # 1 + essentially adds 1 to each dimension = number of dimensions
    return xent_loss + kl_loss

#vae = Model([x, x_c], x_decoded_mean_squash)
vae = Model(x, x_decoded_mean_squash)
vae.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), loss=vae_loss)
vae.summary()

# train the VAE on MNIST digits
#(x_train, _), (x_test, y_test) = mnist.load_data()
#load CUB data
## Generate training, val and test for cub with one-hot encoded labels uncomment for cub split datasets
'''
These images were deleted from the dataset due to having incorrect or corrupt data i.e. the images lack a color channel
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/009.Brewer_Blackbird\Brewer_Blackbird_0028_2682.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/025.Pelagic_Cormorant\Pelagic_Cormorant_0022_23802.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/063.Ivory_Gull\Ivory_Gull_0040_49180.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/063.Ivory_Gull\Ivory_Gull_0085_49456.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/066.Western_Gull\Western_Gull_0002_54825.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/087.Mallard\Mallard_0130_76836.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/093.Clark_Nutcracker\Clark_Nutcracker_0020_85099.jpg  has size  (84, 84)
G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011//images/108.White_necked_Raven\White_Necked_Raven_0070_102645.jpg  has size  (84, 84)
'''
base_dir = 'G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011/'
f = open(base_dir + 'classes.txt')
reader = csv.reader(f, delimiter=' ')
num_classes = 200
train_image_dataset = []
val_image_dataset = []
test_image_dataset = []
train_class_dataset = []
val_class_dataset = []
test_class_dataset = []
for i, row in enumerate(reader):
    image_files = glob.glob(base_dir + '/images/' + row[1] + '/*.jpg')
    if i%2 == 0:
        image_dataset = train_image_dataset
        class_dataset = train_class_dataset
    elif i%4 == 1:
        image_dataset = val_image_dataset
        class_dataset = val_class_dataset
    elif i%4 == 3:
        image_dataset = test_image_dataset
        class_dataset = test_class_dataset
    for image_file in image_files:
        image_resized = cv2.resize(np.asarray(imread(image_file)),
                                   dsize = (84, 84))
        if image_resized.shape != (84, 84, 3):
            print(image_file, " has size ", image_resized.shape)
        image_dataset.append(image_resized)
        class_dataset.append(int(row[0])-1)

print("datasets splits finished splitting")
# print(train_image_dataset)
train_image_dataset = np.array(train_image_dataset)
val_image_dataset = np.array(val_image_dataset)
test_image_dataset = np.array(test_image_dataset)
train_class_dataset = tf.one_hot(np.array(train_class_dataset), num_classes)
val_class_dataset = tf.one_hot(np.array(val_class_dataset), num_classes)
test_class_dataset = tf.one_hot(np.array(test_class_dataset), num_classes)
# print(train_image_dataset)
#print(test_class_dataset.shape())
f.close()

x_train = train_image_dataset.astype('float32') / 255.
#x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
x_test = val_image_dataset.astype('float32') / 255.
#x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)
print(train_class_dataset.shape)

checkpoint = tf.keras.callbacks.ModelCheckpoint('weights.{epoch:03d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=5)
tensorboard_callback = tf.keras.callbacks.TensorBoard('logs/')
def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 5
    if epoch % decay_step == 0 and epoch and lr>5e-7: #stop making it smaller if lr is too small
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr
learning_scheduler= tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

start = time.time()
vae.fit(x_train, x_train,
        shuffle=True,
        # steps_per_epoch=x_train.shape[0] // batch_size,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test),
        callbacks=[tensorboard_callback, checkpoint, learning_scheduler])
# vae.fit([x_train, train_class_dataset], x_train,
#         shuffle=True,
#         epochs=epochs,
#         batch_size=batch_size,
#         steps_per_epoch=x_train.shape[0] // batch_size,
#         validation_data=([x_test, val_class_dataset], x_test),
#         callbacks=[tensorboard_callback, checkpoint, learning_scheduler])

done = time.time()
elapsed = done - start