#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


os.getcwd()


# In[3]:


os.chdir ('C:\\Users\\vikram\\Desktop\\top mentor\\Batch 74 Day 32,7th may 2023\\Project -4 HR Department\\')
os.getcwd()


# In[4]:


employee_df= pd.read_csv('Human_Resources.csv')
display (employee_df)


# In[5]:


display(employee_df.head(5))


# In[6]:


display (employee_df.tail(10))


# In[8]:


employee_df.info()
# 35 features in total, each contains 1470 data points


# In[9]:


employee_df.describe()


# In[10]:


# Replace the 'Attritition' ,'overtime' and 'Over 18' column with integers before performing any visualizations 
employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)


# In[11]:


display (employee_df[['Attrition','OverTime', 'Over18' ]])


# In[12]:


employee_df.isnull().sum()


# In[13]:


sns.heatmap(employee_df.isnull(), yticklabels = False, cbar = False, cmap="Blues")
plt.show()


# In[14]:


employee_df.hist(bins = 30, figsize = (20,50), color = 'r')
plt.show()
# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavy
# It makes sense to drop 'EmployeeCount' and 'Standardhours' since they do not change from one employee to the other


# In[15]:


# It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other
# Let's drop 'EmployeeNumber' as well
employee_df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis=1, inplace=True)


# In[16]:


display (employee_df)


# In[17]:


# display how many employees left the company
employee_df['Attrition'].value_counts()


# In[18]:


left_df        = employee_df[employee_df['Attrition'] == 1]
stayed_df      = employee_df[employee_df['Attrition'] == 0]
display (left_df )
display (stayed_df)


# In[19]:


# Count the number of employees who stayed and left
# It seems that we are dealing with an imbalanced dataset 

print("Total =", len(employee_df))

print("Number of employees who left the company =", len(left_df))
print("Percentage of employees who left the company =", 1.*len(left_df)/len(employee_df)*100.0, "%")
 
print("Number of employees who did not leave the company (stayed) =", len(stayed_df))
print("Percentage of employees who did not leave the company (stayed) =", 1.*len(stayed_df)/len(employee_df)*100.0, "%")


# In[20]:


display (left_df.describe())
display (stayed_df.describe())
# Compare the mean and std of the employees who stayed and left 
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home 
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option level


# In[21]:


correlations = employee_df.corr()
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(correlations, annot = True)
plt.show()

# Job level is strongly correlated with total working hours
# Monthly income is strongly correlated with Job level
# Monthly income is strongly correlated with total working hours
# Age is stongly correlated with monthly income


# In[22]:


plt.figure(figsize=[20,10])
sns.countplot(x = 'Age', hue = 'Attrition', data = employee_df)
plt.show()


# In[23]:


plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x = 'JobRole', hue = 'Attrition', data = employee_df)
plt.subplot(412)
sns.countplot(x = 'MaritalStatus', hue = 'Attrition', data = employee_df)
plt.subplot(413)
sns.countplot(x = 'JobInvolvement', hue = 'Attrition', data = employee_df)
plt.subplot(414)
sns.countplot(x = 'JobLevel', hue = 'Attrition', data = employee_df)

# Single employees tend to leave compared to married and divorced
# Sales Representitives tend to leave compared to any other job 
# Less involved employees tend to leave the company 
# Less experienced (low job level) tend to leave the company


# In[24]:


plt.figure(figsize=[20,20])
plt.subplot(211)
sns.countplot(x = 'DistanceFromHome', hue = 'Attrition', data = employee_df)
plt.show()


# In[25]:


# KDE (Kernel Density Estimate) is used for visualizing the Probability Density of a continuous variable. 
# KDE describes the probability density at different values in a continuous variable. 

plt.figure(figsize=(12,7))

sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Distance From Home')


# In[26]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['YearsWithCurrManager'], label = 'Employees who left', shade = True, color = 'r')
sns.kdeplot(stayed_df['YearsWithCurrManager'], label = 'Employees who Stayed', shade = True, color = 'b')

plt.xlabel('Years With Current Manager')


# In[27]:


plt.figure(figsize=(12,7))

sns.kdeplot(left_df['TotalWorkingYears'], shade = True, label = 'Employees who left', color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], shade = True, label = 'Employees who Stayed', color = 'b')

plt.xlabel('Total Working Years')


# In[28]:


# Box plot Gender vs. Monthly Income
plt.figure(figsize=(15, 10))
sns.boxplot(x = 'MonthlyIncome', y = 'Gender', data = employee_df)


# In[29]:


# Box Plot monthly income vs. job role
plt.figure(figsize=(20, 15))
sns.boxplot(x = 'MonthlyIncome', y = 'JobRole', data = employee_df)


# In[30]:


display (employee_df.head())


# In[31]:


X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
display(X_cat)


# In[32]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
display(X_cat.shape)
display(X_cat)


# In[33]:


X_cat = pd.DataFrame(X_cat)
display (X_cat)


# In[34]:


# Get all numerical columns from the data frame by excluding target variable 'Attrition'
X_numerical = employee_df[['Age', 'DailyRate', 'DistanceFromHome',	'Education', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',	'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	'NumCompaniesWorked',	'OverTime',	'PercentSalaryHike', 'PerformanceRating',	'RelationshipSatisfaction',	'StockOptionLevel',	'TotalWorkingYears'	,'TrainingTimesLastYear'	, 'WorkLifeBalance',	'YearsAtCompany'	,'YearsInCurrentRole', 'YearsSinceLastPromotion',	'YearsWithCurrManager']]
display(X_numerical)


# In[35]:


X_all = pd.concat([X_cat, X_numerical], axis = 1)
display (X_all)


# In[36]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all)


# In[37]:


display (pd.DataFrame(X))


# In[38]:


y = employee_df['Attrition']
display (y)


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
display (X_train.shape)
display (X_test.shape)


# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
display (y_pred)


# In[41]:


from sklearn.metrics import confusion_matrix, classification_report

print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))


# In[42]:


cm = confusion_matrix(y_pred, y_test)
print (cm)


# In[43]:


sns.heatmap(cm, annot=True)


# In[44]:


print(classification_report(y_test, y_pred))


# In[45]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display (y_pred)


# In[46]:


print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))
print ('\n Confusion Matrix')
cm = confusion_matrix(y_pred, y_test)
print (cm)


# In[47]:


sns.heatmap(cm, annot=True)


# In[48]:


print(classification_report(y_test, y_pred))


# In[49]:


print(classification_report(y_test, y_pred))


# In[50]:


from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
display (y_pred)


# In[51]:


print("Accuracy {} %".format( 100 * accuracy_score(y_pred, y_test)))
print ('\n Confusion Matrix')
cm = confusion_matrix(y_pred, y_test)
print (cm)


# In[52]:


sns.heatmap(cm, annot=True)


# In[53]:


print(classification_report(y_test, y_pred))


# In[54]:


get_ipython().system(' pip install tensorflow')


# In[55]:


import tensorflow as tf


# In[56]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=500, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
display (model.summary())


# In[57]:


model.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])


# In[58]:


employee_df['Attrition'].value_counts()


# In[59]:


epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50)


# In[60]:


y_pred = model.predict(X_train)
y_pred = (y_pred > 0.5)
display (y_pred)


# In[61]:


display (epochs_hist.history.keys())


# In[62]:


plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.show()


# In[63]:


plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])
plt.show()


# In[64]:


cm = confusion_matrix(y_train, y_pred)
display (cm)


# In[65]:


sns.heatmap(cm, annot=True)


# In[66]:


print(classification_report(y_train, y_pred))


# In[67]:


y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
display (y_pred)


# In[68]:


cm = confusion_matrix(y_test, y_pred)
display (cm)


# In[69]:


print(classification_report(y_test, y_pred))


# In[ ]:




