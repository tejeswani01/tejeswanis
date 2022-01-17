#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION TASK 3
# ## Exploratory Data Analysis on SampleSuperstore Dataset 
# 
# *For the given dataset (https://bit.ly/3i4rbWl)*
# 
# ## By S.Tejeswani 
# 

# In[153]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[154]:


import warnings
warnings.filterwarnings('ignore')


# ## loading the data

# In[155]:


data = pd.read_csv("C:\\users\\SampleSuperstore.csv")


# In[156]:


data.head() #display first 5 rows


# In[157]:


data.tail() #display last 5 rows


# In[158]:


data.info()


# In[159]:


data.describe()


# In[160]:


data.shape


# In[161]:


data.columns


# In[162]:


data['Ship Mode'].value_counts()


# In[163]:


data['Segment'].value_counts()


# In[164]:


data['Category'].value_counts()


# In[165]:


data['Sub-Category'].value_counts()


# In[166]:


data['State'].value_counts()


# In[167]:


data['Region'].value_counts()


# ## preprocessing the data

# In[168]:


data.isna().any()


# In[169]:


data.isna().sum()


# ## correlation matrix

# In[170]:


data.corr()


# In[171]:


corr=data.corr()
fig,ax=plt.subplots(figsize=(10,10))
sns.heatmap(corr,annot=True)


# ## exploratory data analysis

# In[114]:


sns.pairplot(data,hue='Ship Mode')


# In[73]:


sns.pairplot(data,hue='Segment')


# In[72]:


sns.pairplot(data,hue='Category')


# In[74]:


sns.pairplot(data,hue='Sub-Category')


# In[26]:


sns.countplot(data['Ship Mode'])


# In[27]:


sns.countplot(data['Segment'])


# In[172]:


sns.set(rc={'figure.figsize':(10,10)})
data['Category'].value_counts().plot.pie()


# In[173]:


sns.set(rc={'figure.figsize':(10,10)})
data['Sub-Category'].value_counts().plot.pie()


# In[174]:


sns.set(rc={'figure.figsize':(5,5)})
data['Region'].value_counts().plot.pie()


# In[175]:


plt.figure(figsize=(10,10))
sns.countplot(data['State'])
plt.xticks(rotation=90)
plt.show()


# In[176]:


data.hist(figsize=(10,10),bins=10)
plt.show()


# In[37]:


sns.set(rc={'figure.figsize':(10,5)})
sns.lineplot(x='Quantity',y='Profit',data=data,label='Profit')
plt.show()


# In[122]:


data.groupby('Region')[['Profit','Sales']].sum().plot.bar(color=['blue','green'],figsize=(5,5))
plt.ylabel('Profit and Sales')
plt.show()


# In[123]:


data.groupby('State')[['Profit','Sales']].sum().plot.bar(color=['blue','green'],figsize=(15,15))
plt.ylabel('Profit and Sales')
plt.show()


# In[149]:


data.groupby('Segment')[['Profit','Sales']].sum().plot.bar(color=['blue','green'],figsize=(5,5))
plt.ylabel('Profit and Sales')
plt.show()


# In[150]:


data.groupby('Category')[['Profit','Sales']].sum().plot.bar(color=['blue','green'],figsize=(5,5))
plt.ylabel('Profit and Sales')
plt.show()


# In[177]:


data.groupby('Sub-Category')[['Profit','Sales']].sum().plot.bar(color=['blue','green'],figsize=(5,5))
plt.ylabel('Profit and Sales')
plt.show()


# In[152]:


plt.figure(figsize=(10,10))
sns.barplot(x='Region',y='Sales',data= data, hue='Segment')
plt.show()


# # Observations
# 
# ## ship mode
# ### maximum-standard class
# ### minimum-same day
# 
# ## segment
# ### maximum-consumer
# ### minimum- home office (weak)
# 
# ## category
# ### maximum -office supplies
# ### minimum -technology(weak)
# 
# ## sub category
# ### maximum -binders
# ### minimum -copiers(weak)
# 
# ### high profit- california
# ### less profit-texas(weak)
# 
# ### high profit market-technology
# ### less profit market-furniture(weak)
# 
# ## majority of customers buy quantity of 2 and 3
# 
# ## profit and sales(segments)
# ### maximum consumer segment
# ### minimum home office segment(weak)
# 
# ## profit and sales(regions)
# ### maximum west
# ### minimum-south(weak)
# 
# ### *we need to work on these weak areas to get better profits*

# In[ ]:




