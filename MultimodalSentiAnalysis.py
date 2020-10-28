import os
import sys
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import pickle
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

from itertools import chain
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from sklearn.metrics import recall_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import GRU, MaxPooling1D, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers

from google.colab import drive
drive.mount('/content/gdrive')

data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/EE599 Final Project/final dataset/data_combined.csv',encoding='utf-8')
test_data = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/EE599 Final Project/final dataset/testing_data_3.csv',encoding='utf-8')

data

X = data.iloc[:,1:-1]
y = data['Label']
X

x_test = test_data.iloc[:,1:-1]
y_test = test_data['Label']

test_data

x_train, x_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 0)

MAX_SEQUENCE_LENGTH = 40
MAX_NUM_WORDS = 10000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

embeddings_index = {}

with open("/content/gdrive/My Drive/Colab Notebooks/EE599 Final Project/final dataset/glove.6B.100d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

x_train_text = x_train.iloc[:,193:]
x_val_text = x_val.iloc[:,193:]
x_test_text = x_test.iloc[:,193:]
x_test_text

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)   # get the frequently occuring words
tokenizer.fit_on_texts(x_train_text.Text)           
train_sequences = tokenizer.texts_to_sequences(x_train_text.Text)
val_sequences = tokenizer.texts_to_sequences(x_val_text.Text)
test_sequences = tokenizer.texts_to_sequences(x_test_text.text)

word_index = tokenizer.word_index               # dictionary containing words and their index
print(tokenizer.word_index)                   # print to check
print('Found %s unique tokens.' % len(word_index)) # total words in the corpus
train_text = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
val_text = pad_sequences(val_sequences, maxlen = MAX_SEQUENCE_LENGTH)# get only the top frequent words on train
test_text = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)   # get only the top frequent words on test


scaleable_cols = ['words_count', 'adjective_freq', 'noun_freq', 'adverb_freq', 'verb_freq']

scaler_multicol = MinMaxScaler()
train_multicol_scaled = scaler_multicol.fit_transform(x_train_text[scaleable_cols])
val_multicol_scaled = scaler_multicol.fit_transform(x_val_text[scaleable_cols])
test_multicol_scaled = scaler_multicol.fit_transform(x_test_text[scaleable_cols])

x_train_text = np.hstack((train_text, train_multicol_scaled))
x_val_text = np.hstack((val_text, val_multicol_scaled))
x_test_text= np.hstack((test_text, test_multicol_scaled))


num_words = min(MAX_NUM_WORDS, len(embeddings_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print(x_train_text[1])

x_train_audio = x_train.iloc[:,:193]
x_val_audio = x_val.iloc[:,:193]
x_test_audio = x_test.iloc[:,:193]


print(x_train_audio.shape)
print(x_val_audio.shape)
print(x_test_audio.shape)
print(y_train.shape)
x_train_audio

posts_input = Input(shape=(None,), dtype='int32', name='Text_input')
embedded_posts = Embedding(input_dim=MAX_NUM_WORDS,
                            input_length=MAX_SEQUENCE_LENGTH, 
                            output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=False)(posts_input)

x = layers.GRU(128, activation='relu', return_sequences = True)(embedded_posts)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
x = layers.GRU(64, activation = 'relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
text_model =Model([posts_input], [x])

text_model.summary()

import numpy as np
import os
import sys
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout, Flatten, Embedding
import pickle
from xgboost import XGBClassifier

input1 = Input(shape=(193), name='Audio_input') 
layer = layers.Dense(400,activation ='relu')(input1)
layer = layers.Dropout(0.2)(layer)
layer = layers.Dense(300,activation ='relu')(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Dense(200,activation ='relu')(layer)
layer = layers.Dropout(0.2)(layer)
layer = layers.Dense(100,activation ='relu')(layer)
layer = layers.Dropout(0.2)(layer)


funcmodel = Model([input1], [layer])

funcmodel.summary()

reg_val = 0.0001

combine = layers.concatenate([x,layer])
y = Dense(512,kernel_regularizer = regularizers.l2(reg_val),bias_regularizer = regularizers.l2(reg_val) )(combine)
y = layers.LeakyReLU(alpha=0.1)(y)
y = Dropout(0.2)(y)
y = Dense(128,activation ='relu')(y)
y = Dropout(0.2)(y)



predictions = Dense(3,activation ='softmax')(y) 

combinedModel = Model(inputs=[posts_input,input1 ], outputs=[predictions])
combinedModel.summary()
plot_model(combinedModel, to_file='combined_model.png',show_shapes=True, show_layer_names=True)

callbacks_list = [
                ModelCheckpoint(filepath='model_multi-feature.h5', monitor='val_loss',
                        save_best_only=True,)]
combinedModel.compile(optimizer = 'rmsprop',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])

epochs = 50
batch_size = 64

hist = combinedModel.fit([x_train_text,x_train_audio],  y_train,
                                  epochs = epochs, batch_size = batch_size,
                        callbacks = callbacks_list,
               validation_data = ([x_val_text,x_val_audio],  y_val)).history

import h5py
train_acc = hist['accuracy']
val_acc = hist['val_accuracy']
train_loss = hist['loss']
val_loss = hist['val_loss']

with h5py.File('learningrate_plot.hd5', 'w') as hf:
    hf.create_dataset('train_acc', data = train_acc)
    hf.create_dataset('val_acc', data = val_acc)
    hf.create_dataset('train_loss', data = train_loss)
    hf.create_dataset('val_loss', data = val_loss)

model1 = load_model('model_multi-feature.h5')
pred = model1.predict([x_test_text,x_test_audio])
label_predict = np.argmax(pred,axis=1 )
print(set(label_predict))

label_acc = accuracy_score(label_predict, y_test)

print("ACCURACY : ",label_acc)



with h5py.File('learningrate_plot.hd5', 'r') as hf:
    for i in hf.keys():
        print(i)
    
    train_acc = hf['train_acc'][:] 
    val_acc = hf['val_acc'][:] 
    train_loss = hf['train_loss'][:] 
    val_loss = hf['val_loss'][:] 


epochs = 50

plt.figure()
plt.plot(np.arange(epochs),train_acc,'r',label = "train_acc")
plt.plot(np.arange(epochs),val_acc,'b',label = "val_acc")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.plot(np.arange(epochs),train_loss,'r',label = "train_loss")
plt.plot(np.arange(epochs),val_loss,'b',label = "val_loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

from sklearn.metrics import accuracy_score, confusion_matrix

label7 = ['sadness', 'surprise','neutral','joy', 'anger','disgust','fear']
labels = ['positive', 'negative', 'neutral']
cm = confusion_matrix(y_test, label_predict)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)

fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

