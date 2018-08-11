
# coding: utf-8

# # Recurrent neural networks can also be used as generative models.

# In[18]:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


# In[3]:

# load ascii text and covert to lowercase
filename = "speeches.txt"
raw_text = open(filename,'r',encoding='utf8',errors='ignore').read()
raw_text = raw_text.lower()
raw_text


# In[4]:

# creating set of characters in book and mappinf unique ints with them
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i,c in enumerate(chars))
char_to_int


# In[5]:

n_chars = len(raw_text)
n_vocabs = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocabs)


# In[14]:

seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i+seq_length]
    #print(seq_in)
    seq_out = raw_text[i+seq_length]
    #print(seq_out)
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)


# In[17]:

for i in range(100):
    print(raw_text[i+100])  


# In[12]:

dataY


# ## we must transform the list of input sequences into the form [samples, time steps, features] expected by an LSTM network.

# In[19]:

# reshape X as [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocabs)
# one hot encode out var
y = np_utils.to_categorical(dataY)


# In[ ]:

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1],X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[22]:

# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# In[ ]:

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)


# In[ ]:



