#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
data= pd.read_csv('data.csv')
#data.shape #To find the dimensions of the dataset
data


# In[18]:


#Check for missing values 
# Missing values
data.isnull().sum()
data.isna().sum()


# In[19]:


X = data.iloc[:,2:32].values
Y = data.iloc[:,1].values


# In[20]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])


# In[21]:


#Label encoding 
#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# In[22]:


#Test and train data set 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[23]:


#Feature Scaling 
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[24]:


#Model training 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)


# In[28]:


#Testing your model 
Y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
cm


# In[27]:


#Checking the accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred)
accuracy


# In[ ]:




