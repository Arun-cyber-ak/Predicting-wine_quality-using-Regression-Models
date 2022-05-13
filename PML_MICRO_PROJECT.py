#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv("winequalityN.csv")


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df['fixed acidity'].fillna(df['fixed acidity'].mode()[0], inplace=True)
df['volatile acidity'].fillna(df['volatile acidity'].mode()[0], inplace=True)
df['citric acid'].fillna(df['citric acid'].mode()[0], inplace=True)
df['residual sugar'].fillna(df['residual sugar'].mode()[0], inplace=True)
df['chlorides'].fillna(df['chlorides'].mode()[0], inplace=True)
df['pH'].fillna(df['pH'].mode()[0], inplace=True)
df['sulphates'].fillna(df['sulphates'].mode()[0], inplace=True)


# In[8]:


df.isnull().sum()


# In[9]:


X = df.drop(['type','quality'],axis=1)
y= df['quality']


# In[10]:


#X=df.drop(['type','quality'],axis=1)
#y=df['quality']


# **Splitting the Dataset**

# In[11]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[12]:


from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(X_train,y_train)
y_pred = l.predict(X_test)


# In[13]:


from sklearn.metrics import mean_squared_error,r2_score
m_l = mean_squared_error(y_test,y_pred)
print("Mean Squared error of Linear model is ",m_l)


# In[14]:


r2_l = r2_score(y_test,y_pred)
r2_l


# **Standard scaling**

# In[15]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler
ss = StandardScaler()


# In[16]:


X_train_ss = ss.fit_transform(X_train)


# In[17]:


X_test_ss = ss.transform(X_test)


# ## **Creating the Linear Regression Model**

# In[18]:


from sklearn.linear_model import LinearRegression
ln = LinearRegression()
ln.fit(X_train_ss,y_train)
y_pred_ln_ss = ln.predict(X_test_ss)


# In[19]:


m_ln_ss = mean_squared_error(y_test,y_pred_ln_ss)
print("Mean Squared Error of Linear Model using Standard Scaling",m_ln_ss)


# In[20]:


r2_ln_ss = r2_score(y_test,y_pred_ln_ss)
r2_ln_ss


# ## **Creating the SVR model**

# In[21]:


from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train_ss,y_train)
y_pred_svr_ss = svr.predict(X_test_ss)


# In[22]:


from sklearn.metrics import mean_squared_error,r2_score
m_svr_ss = mean_squared_error(y_test,y_pred_svr_ss)
r2_svr_ss = r2_score(y_test,y_pred_svr_ss)
print("Mean Squared Error of Support Vector Regressor is ",m_svr_ss)


# In[23]:


r2_svr_ss


# ### Creating the SGD Regressor Model

# In[24]:


from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor()
sgd.fit(X_train_ss,y_train)
y_pred_sgd_ss = sgd.predict(X_test_ss)                   


# In[25]:


m_sgd_ss = mean_squared_error(y_test,y_pred_sgd_ss)
print("Mean Squared Error of SGD Regressor is ",m_sgd_ss)


# In[26]:


r2_sgd_ss = r2_score(y_test,y_pred_sgd_ss)
r2_sgd_ss


# ### **Creating the Ridge regression Model**

# In[27]:


from sklearn.linear_model import Ridge
rid = Ridge()
rid.fit(X_train_ss,y_train)
y_pred_rid_ss = rid.predict(X_test_ss)


# In[28]:


m_rid_ss = mean_squared_error(y_test,y_pred_rid_ss)
print("Mean Squared Error of Ridge regression is ",m_rid_ss)


# In[29]:


r2_rid_ss = r2_score(y_test,y_pred_rid_ss)
r2_rid_ss


# ### **Creating the Lasso Regression Model**

# In[30]:


from sklearn.linear_model import Lasso
las = Lasso()
las.fit(X_train_ss,y_train)
y_pred_las_ss = las.predict(X_test_ss)


# In[31]:


m_las_ss = mean_squared_error(y_test,y_pred_las_ss)
print("Mean Squared Error of Lasso Regression is ",m_las_ss)


# In[32]:


r2_las_ss = r2_score(y_test,y_pred_las_ss)
r2_las_ss


# **Calculating the Root Mean Squared Error using Standard Scaling**

# In[33]:


import numpy as np


# In[34]:


print("RMSE of Linear Regression Model: ",np.sqrt(m_ln_ss))

print("RMSE of Support Vector Regressor model: ",np.sqrt(m_svr_ss))

print("RMSE of SGD regressor Model: ",np.sqrt(m_sgd_ss))

print("RMSE of Ridge Regression Model: ",np.sqrt(m_rid_ss))

print("RMSE of Lasso Regression Model: ",np.sqrt(m_las_ss))


# **Finding out the best of Root Mean Squared Error**

# _The Best value of Root Mean Squared error is 0.6662073212886711 and the model is Support Vector Regressor Model_

# In[ ]:




