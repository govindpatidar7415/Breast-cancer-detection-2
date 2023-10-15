#!/usr/bin/env python
# coding: utf-8

# In[1]:


import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pip install sns


# In[3]:


from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()


# In[4]:


cancer_dataset


# In[6]:


type(cancer_dataset)


# In[7]:


cancer_dataset.keys()


# In[8]:


cancer_dataset['data']


# In[9]:


type(cancer_dataset['target'])


# In[10]:


cancer_dataset['target']


# In[11]:


cancer_dataset['target_names']


# In[12]:


print(cancer_dataset['DESCR'])


# In[13]:


print(cancer_dataset['feature_names'])


# In[14]:


print(cancer_dataset['filename'])


# In[15]:


cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],
                        columns = np.append (cancer_dataset['feature_names'],['target']))


# In[16]:


cancer_df.to_csv('breast_cancer_dataframe.csv')


# In[17]:


cancer_df.head(6)


# In[18]:


cancer_df.tail(6)


# In[19]:


cancer_df.info()


# In[20]:


cancer_df.describe()


# In[21]:


cancer_df.isnull().sum()


# In[22]:


sns.pairplot(cancer_df ,hue ='target')


# In[32]:


sns.pairplot(cancer_df,hue ='target',
            vars =['mean radius','mean texture','mean perimeter','mean area','mean smoothness'])


# In[38]:


# Count the target class
sns.countplot(cancer_df['target'])


# In[39]:


# counter plot of feature mean radius
plt.figure(figsize = (20,8))
sns.countplot(cancer_df['mean radius'])


# In[40]:


# heatmap of DataFrame
plt.figure(figsize=(16,9))
sns.heatmap(cancer_df)


# In[41]:


# Heatmap of Correlation matrix of breast cancer DataFrame
plt.figure(figsize=(20,20))
sns.heatmap(cancer_df.corr(), annot = True, cmap ='coolwarm', linewidths=2)


# In[45]:


# create second DataFrame by droping target
cancer_df2 = cancer_df.drop(['target'], axis = 1)
print("The shape of 'cancer_df2' is : ", cancer_df2.shape)

