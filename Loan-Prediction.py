#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import relevant Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn


# In[2]:


#load_data
loan_data = pd.read_csv("https://raw.githubusercontent.com/pradeepiitbhu/loan-prediction-/main/loan-train.csv", on_bad_lines = 'skip')


# In[3]:


loan_data.head()


# In[4]:


#drop unnecessary columns
loan_data.drop("Loan_ID",axis = 1,inplace = True)


# In[5]:


loan_data.shape


# In[6]:


loan_data.columns


# In[7]:


loan_data.isnull().sum()


# In[8]:


loan_data.dtypes


# In[9]:


loan_data.describe(include = "all")


# ## visualising data

# In[10]:


sns.countplot("Gender", data = loan_data , hue = 'Loan_Status')


# In[11]:


sns.countplot("Married", data = loan_data , hue = 'Loan_Status')


# In[12]:


sns.countplot("Dependents", data = loan_data , hue = 'Loan_Status')


# In[13]:


sns.countplot("Education", data = loan_data , hue = 'Loan_Status')


# In[14]:


sns.countplot("Self_Employed", data = loan_data , hue = 'Loan_Status')


# ## filling missing data

# In[15]:


loan_data.Gender.value_counts()


# In[16]:


loan_data.Gender.fillna('Male', inplace = True)


# In[17]:


loan_data.Married.value_counts()


# In[18]:



loan_data.Married.fillna('Yes',inplace = True)


# In[19]:


loan_data.Dependents.value_counts()


# In[20]:


loan_data.Dependents.fillna('0' , inplace = True)


# In[21]:


loan_data.Self_Employed.value_counts()


# In[22]:


loan_data.Self_Employed.fillna('No',inplace = True)


# In[23]:


loan_data.Loan_Amount_Term.value_counts()


# In[24]:


loan_data.Loan_Amount_Term.fillna(360.0,inplace = True)


# In[25]:


loan_data.LoanAmount.fillna(loan_data.LoanAmount.mean(),inplace = True)


# In[26]:


loan_data.Credit_History.unique()


# In[27]:


loan_data.Credit_History.value_counts()


# In[28]:


loan_data.Credit_History.fillna(1.0,inplace = True)


# In[29]:


loan_data.isnull().sum()


# In[30]:


loan_data.head()


# In[31]:


loan_data.isnull().sum()


# ## handling outliers

# In[32]:


sns.boxplot(y = "ApplicantIncome" , data = loan_data)


# In[33]:


loan_data = loan_data[loan_data.ApplicantIncome <=8000]


# In[34]:


sns.boxplot(y = "ApplicantIncome" , data = loan_data)


# In[35]:


sns.boxplot(y = "CoapplicantIncome" , data = loan_data)


# In[36]:


loan_data = loan_data[loan_data.CoapplicantIncome<=6000]


# In[37]:


sns.boxplot(y = "CoapplicantIncome" , data = loan_data)


# ## Encoding the categorical data

# In[38]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[39]:


for i in range(5):
    loan_data.iloc[:,i] = le.fit_transform(loan_data.iloc[:,i])
loan_data["Property_Area"] = le.fit_transform(loan_data["Property_Area"])
loan_data["Loan_Status"] = le.fit_transform(loan_data["Loan_Status"])


# In[40]:


loan_data.head()


# In[41]:


#spliting the traning data
X = loan_data.drop("Loan_Status",axis = 1)


# In[42]:


y = loan_data["Loan_Status"]


# In[43]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[44]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[45]:


X.shape


# ## Split data into train and test

# In[46]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=3)


# # Classification algorithms

# ## Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0)
LR.fit(X_train,y_train)


# In[48]:


#predicting the test set result
y_pred = LR.predict(X_test)


# In[49]:


y_pred


# In[50]:


#measuring accuracy
from sklearn import metrics
print("the accuracy of the LogisticRegression is : ",metrics.accuracy_score(y_pred,y_test))


# In[51]:


#making the confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred , y_test)
sklearn.metrics.plot_confusion_matrix(LR ,X_test,y_test)
plt.show()


# ## Decision Tree

# In[52]:


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_pred


# In[53]:


from sklearn import metrics
print("the accuracy of the Decision Tree is : ",metrics.accuracy_score(y_pred,y_test))


# In[54]:


#making the confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred , y_test)
sklearn.metrics.plot_confusion_matrix(clf ,X_test,y_test)
plt.show()


# ##  Support Vector Machines

# In[55]:


from sklearn.svm import SVC
clf = SVC(kernel = 'linear', random_state = 0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


# In[56]:


from sklearn import metrics
print("the accuracy of the Support Vector Machine model is : ",metrics.accuracy_score(y_pred,y_test))


# In[57]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred , y_test)
sklearn.metrics.plot_confusion_matrix(clf ,X_test,y_test)
plt.show()


# ## K - Nearest Neighbors

# In[58]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
clf.fit(X_train,y_train)


# In[59]:


clf.predict(X_test)


# In[60]:


from sklearn import metrics
print("the accuracy of K-NN model is : ",metrics.accuracy_score(y_pred,y_test))


# In[61]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred , y_test)
sklearn.metrics.plot_confusion_matrix(clf ,X_test,y_test)
plt.show()


# ## Naive Bayes

# In[62]:


from sklearn.naive_bayes import GaussianNB
clf_3 = GaussianNB()
clf_3.fit(X_train,y_train)
clf_3.predict(X_test)


# In[63]:


from sklearn import metrics
print("the accuracy of Naive Byes model is : ",metrics.accuracy_score(y_pred,y_test))


# In[64]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred , y_test)
sklearn.metrics.plot_confusion_matrix(clf ,X_test,y_test)
plt.show()


# In[ ]:




