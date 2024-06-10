#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading the data 


# In[3]:


from sklearn import datasets
 
cancer_data = datasets.load_breast_cancer()
print(cancer_data.data[5])


# In[4]:


#Exploring the data 


# In[5]:


print(cancer_data.data.shape) 
#target set 
print(cancer_data.target)


# In[6]:


#Splitting the data 


# In[7]:


from sklearn.model_selection import train_test_split
 
cancer_data = datasets.load_breast_cancer()
 
X_train, X_test, y_train, y_test = train_test_split(cancer_data.data, cancer_data.target, test_size=0.4,random_state=109)


# In[8]:


#Generating the model


# In[9]:


from sklearn import svm
#create a classifier
cls = svm.SVC(kernel="linear")
#train the model
cls.fit(X_train,y_train)
#predict the response
pred = cls.predict(X_test)


# In[10]:


#Evaluating the model


# In[11]:


from sklearn import metrics
#accuracy
print("acuracy:", metrics.accuracy_score(y_test,y_pred=pred))
#precision score
print("precision:", metrics.precision_score(y_test,y_pred=pred))
#recall score
print("recall" , metrics.recall_score(y_test,y_pred=pred))
print(metrics.classification_report(y_test, y_pred=pred))


# In[12]:


#Character recognition 


# In[13]:


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
#loading the dataset
letters = datasets.load_digits()
#generating the classifier
clf = svm.SVC(gamma=0.001, C=100)
#training the classifier
X,y = letters.data[:-10], letters.target[:-10]
clf.fit(X,y)
#predicting the output 
print(clf.predict(letters.data[:-10]))
plt.imshow(letters.images[6], interpolation='nearest')
plt.show()


# In[ ]:




