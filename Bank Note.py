#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary Libraries :

# In[22]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp


# # Importing the Dataset :

# In[2]:


BankNote = pd.read_csv('BankNote.csv')
BankNote.head()


# In[3]:


BankNote.describe()


# In[4]:


BankNote.info()


# In[5]:


BankNote.shape


# In[6]:


BankNote.isnull().sum()


# In[23]:


pp.ProfileReport(BankNote)


# In[7]:


ind_x = BankNote.drop('class',axis=1)
ind_x.head()


# In[8]:


dep_y = BankNote['class']
dep_y


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(ind_x,dep_y, test_size = 0.2, random_state = 1)


# In[10]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[11]:


from sklearn.preprocessing import StandardScaler
normalize = StandardScaler()

x_train = normalize.fit_transform(x_train)
x_test = normalize.fit_transform(x_test)


# In[12]:


import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Flatten

model = Sequential()
model.add(Dense(4,activation = 'relu'))
model.add(Dense(4,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))


# In[13]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[14]:


model.fit(x_train,y_train,epochs = 50,batch_size = 20)


# In[15]:


y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)


# In[20]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm,annot=True)


# In[17]:


print(classification_report(y_test,y_pred))


# In[18]:


acc = accuracy_score(y_test ,y_pred)
acc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




