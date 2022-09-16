#!/usr/bin/env python
# coding: utf-8

# # Mercedes-Benz Greener Manufacturing
# 
# DESCRIPTION
# 
# Reduce the time a Mercedes-Benz spends on the test bench.
# 
# Problem Statement Scenario:
# Since the first automobile, the Benz Patent Motor Car in 1886, Mercedes-Benz has stood for important automotive innovations. These include the passenger safety cell with a crumple zone, the airbag, and intelligent assistance systems. Mercedes-Benz applies for nearly 2000 patents per year, making the brand the European leader among premium carmakers. Mercedes-Benz is the leader in the premium car industry. With a huge selection of features and options, customers can choose the customized Mercedes-Benz of their dreams.
# 
# To ensure the safety and reliability of every unique car configuration before they hit the road, the company’s engineers have developed a robust testing system. As one of the world’s biggest manufacturers of premium cars, safety and efficiency are paramount on Mercedes-Benz’s production lines. However, optimizing the speed of their testing system for many possible feature combinations is complex and time-consuming without a powerful algorithmic approach.
# 
# You are required to reduce the time that cars spend on the test bench. Others will work with a dataset representing different permutations of features in a Mercedes-Benz car to predict the time it takes to pass testing. Optimal algorithms will contribute to faster testing, resulting in lower carbon dioxide emissions without reducing Mercedes-Benz’s standards.
# 
# Following actions should be performed:
# 
# > If for any column(s), the variance is equal to zero, then you need to remove those variable(s).
# 
# > Check for null and unique values for test and train sets.
# 
# > Apply label encoder.
# 
# > Perform dimensionality reduction.
# 
# > Predict your test_df values using XGBoost.

# In[1]:


# Importing the important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading both Datasets
train_data = pd.read_csv('train.csv')
test_data= pd.read_csv('test.csv')


# Checking/Viewing both data

# In[3]:


train_data.head()


# In[4]:


train_data.shape


# In[5]:


train_data.dtypes


# In[6]:


test_data.head()


# In[7]:


test_data.shape


# In[8]:


test_data.dtypes


# In[9]:


train_data.isna().sum()


# Since there are 378 columns in the train dataset we cannot decide if there are any null values present in the dataset by using the above method so we will apply a different method to check for the null values.

# In[10]:


train_data.isna().sum().any()


# In[11]:


test_data.isna().sum().any()


# Both data consists of some object type data as well and the train data have one extra column available('y'), so it can be considered that we have to perform operations to determine 'y'(Output/Label) and test our model against the test data.
# 
# And the datasets have alot of columns.
# 
# > The data looks as if somekind of encoding has already been applied to hide/secure the features of the cars

# ### If for any column(s), the variance is equal to zero, then you need to remove those variable(s).

# In[12]:


#Creating a copy of our dataset to perform some EDA
train_df=train_data.copy()


# In[13]:


train_df.shape


# In[14]:


train_df.head()


# ### Now since we know that the "ID" column is of no use and the "y" column is the output/label column we can omit these two as well

# In[15]:


train_df.drop('ID',axis=1,inplace=True)


# In[16]:


train_df.shape


# In[17]:


# Removing all those columns who have zero variance and omitting 'y' column from this test
train_df.drop('y',axis=1)
for i in train_df:
    if train_df[i].dtype=='object':
        pass
    elif np.var(train_df[i])==0:
        train_df.drop([train_df[i].name],axis=1,inplace=True)


# In[18]:


train_df.shape


# In[19]:


train_df.head()


# ### Now we have removed all those columns whose varaince was zero and we cannot remove all those object columns as the data itself has some kind of encoding done so it will be nuisance to remove those columns without knowing their significance and the 'y' column is the output/label so we cannot remove that either

#  --------------------------------------------------------------------------------------------------------------------------
# 

# ### Now we are going to split our dataset

# In[20]:


train_data_features = train_df.drop(columns=['y'])
train_data_target = train_df.y
print(train_data_features.shape)
print(train_data_target.shape)


# In[21]:


# Checking the feature and target datasets
train_data_features


# In[22]:


train_data_target


# ### We still have some object data type in our feature dataset, so we need to apply somekind of encoding before moving on to the model part

# In[23]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder() # instaniating the class


# In[24]:


for i in train_data_features:
    if train_data_features[i].dtype=='object':
        print(train_data_features[i].name)
        train_data_features[i]=le.fit_transform(train_data_features[i])
    else:
        pass


# In[25]:


# Now checking the features dataset after encoding
train_data_features


# Checking for correlation of the variables to decide which technique to use further.

# In[39]:


corr=train_data_features.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr[(corr>=0.5)],cmap='viridis',vmax=1.0,vmin=0.4,linewidths=0.1,annot=True,annot_kws={'size':8},square=True)
plt.show()


# #### As the varibales are highly coorelated and the proper expalnation about the variables/features(columns) are also not mentioned for security purpose, we must proceed by trying to keep as much features(columns) intact as possible, so we will have to use any dimesionality reduction technique and since the data is highly correlated and have too many features we are going to use the PCA algorithm. In case of less amount of features/columns we can use the SVD(Singular Value Decomposition)

# In[26]:


from sklearn.decomposition import PCA
pca=PCA(n_components=.95) # we are keeping the 95% of the original data(features)


# In[27]:


pca.fit(train_data_features,train_data_target)


# In[28]:


train_data_features_trans=pca.fit_transform(train_data_features)
print(train_data_features_trans.shape)


# In[29]:


train_data_features_trans


# ### Now, since we have done all the steps and our dataset is ready now we can apply XGBoost

# In[30]:


#!pip install xgboost


# In[31]:


import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error


# In[32]:


train_x,test_x,train_y,test_y=train_test_split(train_data_features_trans,train_data_target,test_size=0.3,random_state=7)
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# ### Using XGBoost we are checking the RMSE

# In[33]:


xgb_reg = xgb.XGBRegressor()
model = xgb_reg.fit(train_x,train_y)
print('RMSE =',np.sqrt(mean_squared_error(model.predict(test_x),test_y)))

