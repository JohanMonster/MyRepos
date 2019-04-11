# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 21:24:18 2019

@author: alpha
"""
import numpy as np
text=open(r"C:\Users\alpha\Downloads\NLTK.txt").read().lower()
char_list=sorted(list(set(text)))
char_to_index={b:a for a,b in enumerate(char_list)}
index_to_char={a:b for a,b in enumerate(char_list)}

len_sentences=60
step_size=3
sentences=[]
sentences_follow=[]
for i in range(0,len(text)-len_sentences,step_size):
    sentences.append(text[i:i+len_sentences])
    sentences_follow.append(text[i+len_sentences])


def vectorize():
    x=np.zeros((len(sentences),60,len(char_list)))
    y=np.zeros((len(sentences),len(char_list)))
    for i,sentence in enumerate(sentences):
        for j,char in enumerate(sentence):
            x[i,j,char_to_index[char]]=1
    for i,char in enumerate(sentences_follow):
        y[i,char_to_index[char]]=1
    return x,y


x,y=vectorize()

#x_train,y_train=x[:33000],y[:33000]
#x_val,y_val=x[33000:],y[33000:]
import tensorflow as tf
#from tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM,Embedding,Bidirectional,ELU,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard


def model():
    input_dim=Input((60,len(char_list)))
    x=Bidirectional(LSTM(128,return_sequences=True,recurrent_dropout=0.6))(input_dim)
    x=(LSTM(64,return_sequences=False,recurrent_dropout=0.6))(x)
    x=Dense(64,kernel_initializer='he_normal',activation=tf.nn.elu)(x)
    x=Dense(len(char_list),activation='softmax')(x)
    model=Model(inputs=input_dim,outputs=x)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
    
model=model()
with tf.device("/gpu:{}".format(1)):
   # callbacks=[]
   # model_checkpoint=ModelCheckpoint(filepath=r'my_model.h5',monitor='val_loss',save_best_only=True)
   # tensorboard=TensorBoard(log_dir='my_log_dir',histogram_freq=1)
   # callbacks.append(model_checkpoint)
   # callbacks.append(tensorboard)
    history=model.fit(x,y,batch_size=128,epochs=40)

model.save('char_level_generator.h5')
model=load_model('char_level_generator.h5')
file_write=open('file_write.txt','w')
file_write.write('')
def predict():
    q=[]
    sample_text=sentences[np.random.randint(0,len(sentences))]
    next_sample=sample_text
    print("Before"+('-'*20))
    print(sample_text)
    for i in range(10000):
        predict=np.zeros((1,60,len(char_list)))
        for i,char in enumerate(next_sample):
            predict[0,i,char_to_index[char]]=1
        z=model.predict(predict)
        index=np.random.choice(range(len(char_list)),p=z.ravel())
        char=index_to_char[index]
        q.append(char)
        next_sample+=char
        #print(sample_text)
        next_sample=next_sample[1:]
    for i in q:
        sample_text+=i
    print("After"+('-'*20))
    print(sample_text)
    file_write.write(sample_text)
    
    
predict()   
file_write.close()

    

