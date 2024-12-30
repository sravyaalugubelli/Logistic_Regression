#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\Bhavya\Downloads\heart (1).csv")


# In[3]:


df


# In[4]:


df.columns


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.isnull()


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated().sum()


# In[13]:


df.drop_duplicates()


# In[14]:


df.info()


# In[15]:


df.corr()


# In[16]:


import seaborn as sns


# In[17]:


sns.heatmap(df.corr(),annot=True,cmap='Blues')


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


log_reg=LogisticRegression()


# In[27]:


log_reg.fit(x_train,y_train)


# In[28]:


y_pred=log_reg.predict(x_test)


# In[29]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[30]:


acc=accuracy_score(y_test,y_pred)


# In[31]:


corr=classification_report(y_test,y_pred)


# In[32]:


con_mat=confusion_matrix(y_test,y_pred)


# In[33]:


acc


# In[34]:


corr


# In[35]:


con_mat


# In[ ]:




