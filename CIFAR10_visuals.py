# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:35:03 2019

@author: alpha
"""

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal

from tensorflow.keras.utils import to_categorical
# %matplotlib inline
import gc

#from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16

class CIFAR10():
    def __init__(self,img_row,img_cols,batch_size,epochs):
        self.img_row=img_row
        self.col=img_row
        self.n_channels=3
        self.batch_size=batch_size
        self.epochs=epochs
        self.train_data=None
        self.train_labels=None
        self.test_data=None
        self.test_labels=None
        self.optim=None
        self.vgg16=None
        self.model=None
        self.callbacks=[]
    def data_loader(self):
        (self.train_data,self.train_labels),(self.test_data,self.test_labels)=cifar10.load_data()
        self.train_data=self.train_data/255.
        self.test_data=self.test_data/255.
        self.train_labels=to_categorical(self.train_labels,num_classes=10)
        self.test_labels=to_categorical(self.test_labels,num_classes=10)
    def model_params(self):
        self.optim=Adam(lr=0.01,decay=1e-3)
    def pretrained_loader(self):
        self.vgg16=VGG16(weights='imagenet',include_top=False,input_shape=(self.img_row,self.col,self.n_channels))
        for layer in self.vgg16.layers:
                if('block5' in layer.name):
                    layer.trainable=True
                else:
                    layer.trainable=False
        print(self.vgg16.summary())
    def base_model(self):
        #inputs=Inputs(self.img_row,self.img_col,self.n_channels)
        self.model_params()
        self.pretrained_loader()
        self.data_loader()
        out_tensor=Flatten()(self.vgg16.output)
        X=Dense(64,kernel_initializer='he_normal',activation=tf.nn.elu)(out_tensor)
        X=Dropout(0.5)(X)
        X=Dense(10,kernel_initializer='he_normal',activation='softmax')(X)
        self.model=Model(self.vgg16.inputs,X)
        self.model.compile(optimizer=self.optim,loss='categorical_crossentropy',metrics=['accuracy'])
    def callback(self):
        es=EarlyStopping(monitor='val_loss',patience=5)
        reduce_lr=ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.3)
        model_check=ModelCheckpoint(monitor='val_loss',save_best_only=True,save_weights_only=True,filepath="my_log_cifar10\model__{epoch:04d}_{val_acc:.4f}.h5")
        tensorboard=TensorBoard(log_dir='my_log_dir',histogram_freq=2)
        self.callbacks=[es,reduce_lr,model_check,tensorboard]
    def fit(self):
        self.callback()
        self.base_model()
        self.model.fit(self.train_data,self.train_labels,epochs=self.epochs,batch_size=self.batch_size,callbacks=self.callbacks,validation_data=(self.test_data,self.test_labels))
        print(self.model.evaluate(self.test_data,self.test_labels))
gc.collect()
    
obj=CIFAR10(32,32,64,20)


obj.fit()
   
        
        
