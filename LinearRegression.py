#!/usr/bin/env python
# coding: utf-8

# In[43]:


# imports of useful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# sklearn comes with anaconda installation
from sklearn import linear_model


# In[44]:


# load data in pandas dataframe
df = pd.read_csv("DatasetLinearPolynomial.csv")
df


# In[45]:


# plot a scatter plot to get idea on distribution of data plane
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('time')
plt.ylabel('consumption')
plt.scatter(df.time, df.consumption, color='red', marker='+')


# In[46]:


# create linear regression object
reg = linear_model.LinearRegression()

# will fit the data and train the linear regression model using available data points
# used values because it will contain values without feature names
reg.fit(df[['time']].values, df.consumption.values)


# In[47]:


# predict the energy consumption at time 26
reg.predict([[26]])


# In[48]:


# plot a scatter plot with dataframes time to predict the consumption and plotting on y axis
get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('time')
plt.ylabel('consumption')
plt.scatter(df.time, df.consumption, color='red', marker='+')
plt.plot(df.time, reg.predict(df[['time']].values), color = 'blue')

