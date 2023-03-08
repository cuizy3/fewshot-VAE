import os
import csv
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tf_slim as slim
import time
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from imageio import imread, imwrite

# load test/validation data from cub-200, mini-ImageNet, tiered ImageNet
# n-way k-shot classification: select n classes and then k examples from those n classes to metalearn from
# evaluate model on all examples that were not part of the k examples
# repeat this many times to achieve some statistical confidence interval

# CUB 200 has 200 classes. We will train on 100 of them and leave 50 for metavalidation and 50 for metatest
# MiniImageNet has 100 classes. We will train on 64 of them and leave 16 for metavalidation and 20 for metatest
# TieredImageNet has 34 super-classes split into 20, 6 and 8 disjoint sets of training, validation and test

# cub_200_data = tfds.load(name='caltech_birds2011', shuffle_files=False)
# print(cub_200_data)

## Generate class csvs
class_count = 0
def generate_class_csv(data_split):
    global class_count
    assert(data_split in ['train', 'val', 'test'])
    base_dir = 'G:/meta_learning/datasets/mini_Imagenet/'
    f = open(base_dir + data_split + '.csv')
    reader = csv.reader(f)
    fw = open(base_dir + data_split + '_classes.csv', 'w', newline = '')
    reader = csv.reader(f)
    writer = csv.writer(fw)
    class_set = set()
    next(reader)
    for row in reader:
        if row[1] not in class_set:
            class_set.add(row[1])
            writer.writerow([class_count, row[1]])
            class_count += 1
    fw.close()
    f.close()

# generate_class_csv('train')
# generate_class_csv('val')
# generate_class_csv('test')
# print(class_count)

## Generate training dataset for miniImagenet
def generate_miniImagenet_datasplit(data_split):
    assert(data_split in ['train', 'val', 'test'])
    base_dir = 'G:/meta_learning/datasets/mini_Imagenet/'
    f = open(base_dir + data_split + '_classes.csv')
    reader = csv.reader(f)
    dataset = []
    for row in reader:
        image_files = glob.glob(base_dir + row[1] + '/*.jpeg')
        for image_file in image_files:
            image_resized = cv2.resize(np.asarray(imread(image_file)),
                                       dsize = (84, 84))
            dataset.append([image_resized, tf.one_hot(int(row[0]), 100)])
    f.close()
    return dataset

train_dataset = generate_miniImagenet_datasplit('train')
val_dataset = generate_miniImagenet_datasplit('val')
test_dataset = generate_miniImagenet_datasplit('test')

## Generate training, val and test for cub with one-hot encoded labels uncomment for cub split datasets
# base_dir = 'G:/meta_learning/datasets/CUB_200_2011/CUB_200_2011/'
# f = open(base_dir + 'classes.txt')
# reader = csv.reader(f, delimiter=' ')
# num_classes = 200
# train_dataset = []
# val_dataset = []
# test_dataset = []
# for i, row in enumerate(reader):
#     image_files = glob.glob(base_dir + '/images/' + row[1] + '/*.jpg')
#     if i%2 == 0:
#         dataset = train_dataset
#     elif i%4 == 1:
#         dataset = val_dataset
#     elif i%4 == 3:
#         dataset = test_dataset
#     for image_file in image_files:
#         image_resized = cv2.resize(np.asarray(imread(image_file)),
#                                    dsize = (84, 84))
#         dataset.append([image_resized, tf.one_hot(int(row[0])-1, num_classes)])
#         print(dataset)
#         break
#     break
# print(len(test_dataset))
# f.close()