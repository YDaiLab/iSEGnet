import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, recall_score, roc_auc_score, r2_score
import numpy as np
import os
import os.path
import sys
import csv
import time
import json
import argparse
import pandas as pd
from os import path
import random as rd
import h5py
import gc

kernal_num1 = int(sys.argv[1])
kernal_num2 = kernal_num1 * 2
kernal_size = int(sys.argv[2])
L2 = float(sys.argv[3])
drop_rate = float(sys.argv[4])

h5d_file_path = "/Shang_PHD/Shang/Shang/July_2021/A549/Model/region_promoter_500_500/Model/model_regression.h5"
label_data = pd.read_csv("/Shang_PHD/Shang/Shang/July_2021/A549/Data/A549_RNAseq_regression.txt", delimiter="\t")
with h5py.File('/Shang_PHD/Shang/Shang/July_2021/A549/Data/h5file/epi_data.h5', 'r') as hf:
    epigenetics_input_data = hf['dataset_1'][:]

with h5py.File('/Shang_PHD/Shang/Shang/July_2021/A549/Data/h5file/seq_data.h5', 'r') as hf:
    sequence_input_data = hf['dataset_1'][:]
    
label_data = label_data.values

### Select region 2000 TSS 500
DNase_index = np.array(range(4000, 6500))
H3K4me1_index = np.array(range(11000, 13500))
H3K4me3_index = np.array(range(18000, 20500))
H3K9me3_index = np.array(range(25000, 27500))
H3K27me3_index = np.array(range(32000, 34500))
H3K36me3_index = np.array(range(39000, 41500))
WGBS_index = np.array(range(46000, 48500))
epi_index = np.concatenate((DNase_index, H3K4me1_index, H3K4me3_index,
                            H3K9me3_index, H3K27me3_index, H3K36me3_index,WGBS_index), axis = 0)
seq_index = np.array(range(16000, 26000))

epigenetics_input_data = epigenetics_input_data[:, epi_index]
sequence_input_data = sequence_input_data[:, seq_index]

All_data = np.concatenate((epigenetics_input_data, sequence_input_data), axis = 1)
All_data = np.concatenate((All_data, label_data), axis = 1)

random_oder_path = "/Shang_PHD/Shang/Shang/July_2021/A549/Model/random_order_path.txt"

if os.path.isfile(random_oder_path):
    random_order = pd.read_csv(random_oder_path, delimiter="\t", header = None)[0].values.tolist()
else:
    random_order = range(0, All_data.shape[0])
    random_order = list(random_order)
    rd.shuffle(random_order)
    rd.shuffle(random_order)
    np.savetxt(random_oder_path, random_order, fmt='%i')

All_data = All_data[random_order]

All_data = All_data[All_data[:, All_data.shape[1]-1] != 0, :]

del epigenetics_input_data
del sequence_input_data
gc.collect()

train_data_epi = All_data[0:10004, 0:17500]
train_data_seq = All_data[0:10004, 17500:27500]
train_data_label = All_data[0:10004, 27500:27501]
valid_data_epi = All_data[10004:13339, 0:17500]
valid_data_seq = All_data[10004:13339, 17500:27500]
valid_data_label = All_data[10004:13339, 27500:27501]
test_data_epi = All_data[13339:, 0:17500]
test_data_seq = All_data[13339:, 17500:27500]
test_data_label = All_data[13339:, 27500:27501]

epi_input = keras.Input(shape = (17500,), name="epi_input", dtype=tf.float32)
epi_input_reshape = layers.Reshape(target_shape = (7, 2500, 1))(epi_input)
epi_input_reshape_transpose = layers.Permute((2, 1, 3))(epi_input_reshape)
epi_input_padding = layers.ZeroPadding2D(padding = ((10, 9), (0, 0)))(epi_input_reshape_transpose)
epi_conv = layers.Conv2D(filters=kernal_num1, kernel_size=(kernal_size, 7),
                         activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2),
                         padding = "valid", use_bias = False)(epi_input_padding)
epi_conv_bn = layers.BatchNormalization()(epi_conv)
epi_conv_act = layers.Activation('relu')(epi_conv_bn)
epi_conv_dp = layers.Dropout(drop_rate)(epi_conv_act)
epi_input_padding_l2 = layers.ZeroPadding2D(padding = ((5, 4), (0, 0)))(epi_conv_dp)
epi_conv_l2 = layers.Conv2D(filters=kernal_num2, kernel_size=(kernal_size, 1),
                         activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2),
                         padding = "valid")(epi_input_padding_l2)
epi_conv_bn_l2 = layers.BatchNormalization()(epi_conv_l2)
epi_conv_act_l2 = layers.Activation('relu')(epi_conv_bn_l2)
epi_conv_dp_l2 = layers.Dropout(drop_rate)(epi_conv_act_l2)
epi_pool_l2 = layers.AveragePooling2D(pool_size=(5,1), strides = 5, padding = "valid")(epi_conv_dp_l2)
seq_input = keras.Input(shape = (10000,), name="seq_input", dtype=tf.float32)
seq_input_reshape = layers.Reshape(target_shape = (2500, 4, 1))(seq_input)
#seq_input_reshape_transpose = layers.Permute((2, 1, 3))(seq_input_reshape)
seq_input_padding = layers.ZeroPadding2D(padding = ((10, 9), (0, 0)))(seq_input_reshape)
seq_conv = layers.Conv2D(filters=kernal_num1,kernel_size=(kernal_size, 4), activation=None, 
                         kernel_regularizer=tf.keras.regularizers.l2(L2), 
                         padding = "valid", use_bias = False)(seq_input_padding)
seq_conv_bn = layers.BatchNormalization()(seq_conv)
seq_conv_act = layers.Activation('relu')(seq_conv_bn)
seq_conv_dp = layers.Dropout(drop_rate)(seq_conv_act)
seq_input_padding_l2 = layers.ZeroPadding2D(padding = ((5, 4), (0, 0)))(seq_conv_dp)
seq_conv_l2 = layers.Conv2D(filters=kernal_num2, kernel_size=(kernal_size, 1),
                         activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2),
                         padding = "valid")(seq_input_padding_l2)
seq_conv_bn_l2 = layers.BatchNormalization()(seq_conv_l2)
seq_conv_act_l2 = layers.Activation('relu')(seq_conv_bn_l2)
seq_conv_dp_l2 = layers.Dropout(drop_rate)(seq_conv_act_l2)
seq_pool_l2 = layers.AveragePooling2D(pool_size=(5,1), strides = 5, padding = "valid")(seq_conv_dp_l2)
merged_input = tf.concat([epi_pool_l2,seq_pool_l2], 2)
merged_input_padding = layers.ZeroPadding2D(padding = ((1, 0), (0, 0)))(merged_input)
merged_conv = layers.Conv2D(filters=126, kernel_size=(2, 2), 
                            activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2),
                            padding = "valid")(merged_input_padding)
merged_conv_bn = layers.BatchNormalization()(merged_conv)
merged_conv_act = layers.Activation('relu')(merged_conv_bn)
merged_conv_dp = layers.Dropout(drop_rate)(merged_conv_act)
merged_conv_l2 = layers.Conv2D(filters=64, kernel_size=(1, 1), 
                            activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2),
                            padding = "valid")(merged_conv_dp)
merged_conv_bn_l2 = layers.BatchNormalization()(merged_conv_l2)
merged_conv_act_l2 = layers.Activation('relu')(merged_conv_bn_l2)
merged_conv_dp_l2 = layers.Dropout(drop_rate)(merged_conv_act_l2)
merged_pool_l2 = layers.AveragePooling2D(pool_size=(5,1), strides = 5, padding = "valid")(merged_conv_dp_l2)
flat = layers.Flatten()(merged_pool_l2)
dense1 = layers.Dense(256, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(flat)
dense1_bn = layers.BatchNormalization()(dense1)
dense1_act = layers.Activation('relu')(dense1_bn)
dense1_dp = layers.Dropout(drop_rate)(dense1_act)
dense2 = layers.Dense(64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(dense1_dp)
dense2_bn = layers.BatchNormalization()(dense2)
dense2_act = layers.Activation('relu')(dense2_bn)
dense2_dp = layers.Dropout(drop_rate)(dense2_act)
dense3 = layers.Dense(16, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(dense2_dp)
dense3_bn = layers.BatchNormalization()(dense3)
dense3_act = layers.Activation('relu')(dense3_bn)
output_d = layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(dense3_act)
output = layers.Activation('relu', name= 'expression')(output_d)
model = keras.Model(inputs=[epi_input, seq_input], outputs=output)

model.compile(loss=['mse'], 
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics={"expression" : tfa.metrics.RSquare(y_shape=(1,))})
checkpoint = keras.callbacks.ModelCheckpoint(h5d_file_path, monitor='val_r_square', 
                             verbose=1, save_best_only=True, mode='max')
es = keras.callbacks.EarlyStopping(monitor='val_r_square', mode='max', verbose=1, patience=10)

model.fit(x = [train_data_epi, train_data_seq],
          y = train_data_label,
          validation_data = ([valid_data_epi, valid_data_seq], valid_data_label),
          batch_size = 100,
          epochs=100, callbacks=[es,checkpoint])

model.load_weights(h5d_file_path)

expression_gene_prediction = model.predict(x = [test_data_epi, test_data_seq])

R2_value = [r2_score(test_data_label, expression_gene_prediction)]

f=open('/Shang_PHD/Shang/Shang/July_2021/A549/Model/promoter_hypersearching_value.txt','a')
np.savetxt(f, R2_value, fmt='%1.5f', newline=", ")
f.write("\n")
f.close()

import scipy.stats

pearson = [scipy.stats.pearsonr(test_data_label.flatten(), expression_gene_prediction.flatten())]

f=open('/Shang_PHD/Shang/Shang/July_2021/A549/Model/promoter_hypersearching_value.txt','a')
np.savetxt(f, pearson[0], fmt='%1.5f', newline=", ")
f.write("\n")
f.close()

