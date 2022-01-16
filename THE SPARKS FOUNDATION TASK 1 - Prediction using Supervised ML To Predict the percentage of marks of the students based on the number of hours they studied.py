#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION TASK 1
# ## PREDICTION USING SUPERVISED ML 
# *For the given dataset (http://bit.ly/w-data)*
# ## By S.Tejeswani 
# 
# ### PREDICT THE PERCENTAGE OF A STUDENT BASED ON NUMBER OF STUDY HOURS
# ### PREDICTED SCORE IF THE STUDENT STUDIES FOR 9.25 HOURS/DAY

# ### importing required libraries

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ### reading the data

# In[37]:


data = pd.read_csv('http://bit.ly/w-data')
data.head(5)


# In[38]:


data.isnull == True 


# ### plotting the data

# In[39]:


sns.set_style('whitegrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[40]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# ### defining and splitting the data

# In[41]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# ### training the algorithm

# In[42]:


#as we have split the data we need to train it
regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# ### plotting the regression line

# In[43]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# ### score comparision

# In[44]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[45]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# ### accuracy of the model

# In[46]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# ### PREDICTED SCORE IF THE STUDENT STUDIES FOR 9.25 HOURS/DAY

# In[47]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],3)))

