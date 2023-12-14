#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing all essential liabraries
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import set_config 
from sklearn.impute import SimpleImputer 
#from category_encoders import TargetEncoder, LeaveOneOutEncoder
from bs4 import BeautifulSoup as bs 
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Importing the data and checking features

# In[3]:


data = pd.read_csv(r"C:\Users\hp\Downloads\credit_risk_dataset.csv")
data.head()


# In[4]:


#checking the shape of  data
data.shape


# In[5]:


## checking all statistics measures
data.describe().T.sort_values(ascending = 0,by = "mean").style.background_gradient(cmap = "BuGn")\
.bar(subset = ["std"], color ="red").bar(subset = ["mean"], color ="blue")


# In[6]:


#checking the features of data
data.columns


# Checking for missing Values

# In[7]:


data.isnull().sum()


# Imputing Missing values with mean because of numerical columns

# In[8]:


data['person_emp_length'].fillna(value = data['person_emp_length'].mean() , inplace = True)
data['loan_int_rate'].fillna(value = data['loan_int_rate'].mean() , inplace = True)


# In[9]:


data.isnull().sum()


# checking for duplicate values

# In[10]:


data.duplicated().sum()


# In[11]:


data.drop_duplicates(keep='last' , inplace = True)
data.duplicated().sum()


# Rearrange the columns name because we want target variable in last 

# In[12]:


data = data[['person_age', 'person_income', 'person_home_ownership',
       'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income',
       'cb_person_default_on_file', 'cb_person_cred_hist_length','loan_status']]


# In[13]:


data.head()


# # Univariate Analysis

# In[14]:


sns.boxplot(data['person_age'])


# In[15]:


# calculating quantile
percentile25 = data['person_age'].quantile(0.25)
percentile75 = data['person_age'].quantile(0.75)
iqr = (percentile75 - percentile25)
#Finding the upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding outliers
data[data['person_age'] > upper_limit]
data[data['person_age'] < lower_limit]
#treating outliers
data = data[data['person_age'] < upper_limit]
data.shape


# In[17]:


sns.boxplot(data['person_age'])


# In[18]:


sns.boxplot(data['person_income'])


# In[19]:


# calculating quantile
percentile25 = data['person_income'].quantile(0.25)
percentile75 = data['person_income'].quantile(0.75)
iqr = (percentile75 - percentile25)
#Finding the upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding outliers
data[data['person_income'] > upper_limit]
data[data['person_income'] < lower_limit]
#treating outliers
data['person_income'] = np.log(data['person_income'])


# In[20]:


data['person_income'] = np.log(data['person_income'])


# In[21]:


data['person_income']


# In[22]:


sns.boxplot(data['person_income'])


# In[23]:


sns.boxplot(data['person_emp_length'])


# In[24]:


sns.boxplot(data['loan_amnt'])


# In[25]:


# calculating quantile
percentile25 = data['loan_amnt'].quantile(0.25)
percentile75 = data['loan_amnt'].quantile(0.75)
iqr = (percentile75 - percentile25)
#Finding the upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding outliers
data[data['loan_amnt'] > upper_limit]
data[data['loan_amnt'] < lower_limit]
#treating outliers
data['loan_amnt'] = np.log(data['loan_amnt'])


# In[26]:


sns.boxplot(data['loan_amnt'])


# In[27]:


sns.boxplot(data['loan_int_rate'])


# In[28]:


# calculating quantile
percentile25 = data['loan_int_rate'].quantile(0.25)
percentile75 = data['loan_int_rate'].quantile(0.75)
iqr = (percentile75 - percentile25)
#Finding the upper and lower limits
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
#Finding outliers
data[data['loan_int_rate'] > upper_limit]
data[data['loan_int_rate'] < lower_limit]
#treating outliers
data['loan_int_rate'] = np.log(data['loan_int_rate'])


# In[29]:


sns.boxplot(data['loan_int_rate'])


# In[30]:


sns.boxplot(data['loan_percent_income'])


# In[ ]:





# In[31]:


data["loan_percent_income"]


# In[32]:


sns.boxplot(data['loan_percent_income'])


# In[33]:


sns.boxplot(data['cb_person_cred_hist_length'])


# In[34]:


data.isnull().sum()


# In[35]:


data['person_emp_length'].fillna(value = data['person_emp_length'].mean() , inplace = True)


# In[36]:


data.shape


# In[38]:


data.person_age.plot(kind='kde')


# from above distribution plot, we can see that mostly persons are in the age group of 25-40

# In[40]:


#checking age distribution
plt.hist(data['person_age'] , color ='green',)


# From above histogram , we can clearly see that persons of age group 20-40 are higher demading for loan

# In[41]:


data.person_income.plot(kind='kde')


# In[42]:


#countplot for person income
data['person_income'].plot(kind='hist')


# In[43]:


## Checking for the population's house type through pie chart
Data1 = data["person_home_ownership"].value_counts()
Data1.plot.pie(autopct='%.0f%%')
plt.title("Person Home Ownership")
plt.legend(fontsize=5)
plt.show()


# From the above pie chart , we can see that 50% persons are on rented , 41% are on mortgage , 8% or persons are having own house 
# 

# In[44]:


data['loan_intent'].value_counts()


# In[45]:


# checking for the type of loan requirement 
plt.hist(data['loan_intent'] , color ='maroon',)


# From the above histogram , we can see the loan requirements like 
# EDUCATION            6453
# MEDICAL              6071
# VENTURE              5719
# PERSONAL             5521
# DEBTCONSOLIDATION    5212
# HOMEIMPROVEMENT      3605

# In[46]:


#checking  the loan grade through histogram
plt.hist(data['loan_grade'] , color ='green',)


# From the above histogram , we can see that loan grade A and B are mostly in demand

# In[47]:


# checking the loan amount distribution through histogram
plt.hist(data['loan_amnt'] , color ='green',)


# From the above histogram , we can see that loan amount of 5000 to 10000 are in demand

# In[48]:


data["loan_percent_income"].value_counts()


# In[49]:


#checking the loan percent income distribution
sns.countplot(x = "loan_percent_income", data=data)
plt.title("Loan Percent Income")

plt.show()


# From the above countplot, we can see that loan percent income of 0.10 is most

# In[50]:


#checking cibil default person through pie chart 
Data2 = data["cb_person_default_on_file"].value_counts()
Data2.plot.pie(autopct='%.0f%%')
plt.title("cb person default on file")
plt.legend(fontsize=5)
plt.show()


# From the above pie chart , We can see that 82% is not having default cibil 

# In[51]:


#checking the credit history length through countplot
sns.countplot(x = "cb_person_cred_hist_length", data=data)
plt.title("Credit History Length")
plt.show()


# From the above plot , we can clearly see that credit history length of 2 , 3 , 4 are highest in persons

# In[52]:


#checking the loan status through pie chart 
Data3 = data["loan_status"].value_counts()
Data3.plot.pie(autopct='%.0f%%')
plt.title("Loan Status")
plt.legend(fontsize=5)
plt.show()


# From the above pie chart , we can see that only 22 percent persons are getting loan

# In[425]:


#BIvariate


# In[53]:


sns.scatterplot(y = data["loan_amnt"], x= data['person_income'], hue = data["loan_status"])


# This plot shows people who have low income they have loan approved

# In[54]:


plt.figure(figsize=(8, 6))
sns.countplot(x='loan_grade', hue='loan_status', data=data, palette='Set2')
plt.title('Bar Plot of Loan Grade by Default Status')
plt.xlabel('Loan Grade')
plt.ylabel('Count')
plt.show()


# this bar plot showing that most of the people who have A & B grades loan are get rejected

# In[55]:


plt.figure(figsize=(8, 6))
sns.boxplot(x='loan_status', y='person_age', data=data, palette='Set2')
plt.title('Box Plot of Default Status vs. Person Age')
plt.xlabel('Loan Status')
plt.ylabel('Person Age')
plt.show()


# by this , we can say that people who have old age, then loan get rejected for them

# In[56]:


sns.heatmap(data.corr() , annot = True , cmap = "Set3" , linecolor = "black" , linewidth = 4)
plt.show()


# by this heat map , we can say that 'person_age' & 'credit_hist_length' has strong positive correlation, and
# 'loan_int_rate' & 'loan_status' has weak correlation

# # SPRINT 2 - Data Preparation and Model Building
# 

# Problem Statement - Given various features about a customer like Age, Income, Loan Amount, Loan Intent, Home Ownership etc.. , predict if in case the loan is given, will the customer default or not on the Loan payments.

# Step - 1: Load the data
# 

# In[57]:


Data = pd.read_csv(r"C:\Users\hp\Downloads\credit_risk_dataset.csv")
Data.head()


# Step - 2: Document the below mentioned points properly: 
# - the input variables => 'person_age', 'person_income', 'person_home_ownership',
#        'person_emp_length', 'loan_intent', 'loan_grade', 'loan_amnt',
#        'loan_int_rate', 'loan_percent_income',
#        'cb_person_default_on_file', 'cb_person_cred_hist_length',
# - the target variable = > loan_status
# - the type of ML Task => Classification
# - Identify the Evaluation Metric.
# 	- For regression task - Mean Absolute Error
# 	- For classification task - Accuracy
# 

# from sklearn.model_selection import train_test_split 
# from sklearn.linear_model import LinearRegression 
# from sklearn.preprocessing import StandardScaler,MinMaxScaler 
# from sklearn.tree import DecisionTreeRegressor 
# from sklearn.ensemble import RandomForestRegressor 
# from sklearn.pipeline import Pipeline, make_pipeline 
# from sklearn.compose import ColumnTransformer 
# from sklearn import set_config 
# from sklearn.impute import SimpleImputer 
# from category_encoders import TargetEncoder, LeaveOneOutEncoder
# from bs4 import BeautifulSoup as bs 
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline, make_pipeline

# In[58]:


data.head()


# In[59]:


data.isnull().sum()


# In[60]:


#checking the data information
data.info()


# In[61]:


cat_cols = ['person_home_ownership','loan_intent','loan_grade','cb_person_default_on_file']
num_cols = ['person_age', 'person_income','person_emp_length','loan_amnt','loan_int_rate','loan_percent_income','cb_person_cred_hist_length']


# Splitting the data in training and testing 

# In[62]:


X_train, X_test, Y_train, Y_test = train_test_split(data.drop(['loan_status'], axis=1), data['loan_status'],\
                                                   test_size=0.3, random_state=100)


# Now we have for parts of data as X_train, X_test, Y_train, Y_test
# We will use X_train , Y_train for traing the module and X_test for testing and Y_test for checking  accuracy

# num_pipeline = Pipeline(steps=[('Standard Scaler',StandardScaler()),('Imputer',SimpleImputer(strategy='mean'))])
# cat_pipeline = Pipeline(steps=[('OneHot Encoder',OneHotEncoder(drop='first'))])
# ct = ColumnTransformer([('Standardization',num_pipeline,num_cols),\
#                              ('OneHotEncoder',cat_pipeline, cat_cols)])
# lr = LinearRegression()
# pipe = make_pipeline(ct,lr)

# In[63]:


X_train


# # Data Preprocessing on Train data

# Dividing the training data into categorical features and numerical features

# In[64]:


X_train_obj = X_train.select_dtypes(['object'])


# In[65]:


X_train_numeric = X_train.select_dtypes(['int','float'])


# In[66]:


#making an object for the class of onehot encoder
oe = OneHotEncoder(drop='first', sparse_output=False)


# In[67]:


#applying the onehotencoder to categorical features
pd.DataFrame(oe.fit_transform(X_train_obj), index=X_train_obj.index, columns=oe.get_feature_names_out())


# In[68]:


X_train_obj_oe = pd.DataFrame(oe.fit_transform(X_train_obj), index=X_train_obj.index,\
            columns=oe.get_feature_names_out(X_train_obj.columns))


# In[69]:


X_train_obj_oe


# In[70]:


#setting output in pandas form
set_config(transform_output='pandas')


# creating object for standardization and normalization 

# In[71]:


sc = StandardScaler()
mx = MinMaxScaler()


# In[72]:


data.replace([np.inf, -np.inf], 0, inplace=True)


# In[73]:


#Applying standardization on numerical features
X_train_num_std = sc.fit_transform(X_train_numeric)


# In[74]:


#np.isnan(data.any()) #and gets False
np.isfinite(data.all()) #and gets True


# In[75]:


X_train_num_std


# Merging the both data frame along with columns

# In[76]:


X_train_processed = X_train_num_std.merge(X_train_obj_oe, left_index=True, right_index=True)


# In[77]:


X_train_processed


# # Data Preprocessing on Test Data

# In[78]:


X_test_obj = X_test.select_dtypes(['object'])
X_test_num = X_test.select_dtypes(['int','float'])


# In[79]:


X_test_obj_oe = oe.transform(X_test_obj)
X_test_num_std = sc.transform(X_test_num)


# In[80]:


#Merging the both categorical and numerical features along with columns
X_test_processed = X_test_num_std.merge(X_test_obj_oe, left_index=True, right_index=True)


# # Create  Modules

# Creating ML Modules using different algorithms

# In[81]:


#model using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train_processed, Y_train)


# In[82]:


# Model using LogisticRegression
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train_processed, Y_train)


# In[83]:


#Model using SVC
from sklearn.svm import SVC
sv_classifier = SVC(kernel = 'rbf' , gamma = 0.5 , C = 1.0)
sv_classifier.fit(X_train_processed, Y_train)


# In[84]:


# Model usinng RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators = 100 , random_state = 0)
random_forest_classifier.fit(X_train_processed, Y_train)


# In[85]:


# Model using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_processed, Y_train)


# # Testing on Models

# In[86]:


#Testing using KNN
y_pred_knn = knn.predict(X_test_processed)
y_pred_knn


# In[87]:


# Testing using LogisticRegression
lr_pred = logistic_regression.predict(X_test_processed)
lr_pred


# In[88]:


#Testing using SVC
svc_pred = sv_classifier.predict(X_test_processed)
svc_pred


# In[89]:


#Testing using RandomForest Classifier
y_test_pred_rf = random_forest_classifier.predict(X_test_processed)
y_test_pred_rf 


# In[90]:


#Testing using DecisionTreeClassifier
y_test_pred_dt = dt_classifier.predict(X_test_processed)
y_test_pred_dt


# # Evolution metrics

# In[91]:


#Accuracy score in KNN
metrics.accuracy_score(Y_test , y_pred_knn )


# In[92]:


#Accuracy score in logisticRegression
metrics.accuracy_score(Y_test , lr_pred )


# In[93]:


#Accuracy score in SVC
metrics.accuracy_score(Y_test , svc_pred )


# In[94]:


#Accuracy score in RandomForestClassifier
metrics.accuracy_score(Y_test , y_test_pred_rf )


# In[95]:


#Accuracy score in DecisionTreeClassifier
metrics.accuracy_score(Y_test , y_test_pred_dt )


# In[ ]:





# In[97]:


data = {'Algorithms': ['KNN' , 'Logistic Regression','SVM', 'Random Forest Classifier' , 'Decision Tree'],
        'Accuracy': [89.32 , 86.61 , 91.35 , 93.04 , 88.32]}
plt.figure(figsize=(12, 6))
sns.barplot(x='Accuracy', y='Algorithms', data=data, color='blue')
plt.xlabel('Accuracy in %age')
plt.ylabel('Algorithms')
plt.title('Algorithms Accuracies')
plt.xlim(80, 95)  # Adjust this range based on your actual accuracies
plt.show()


# # Randam Forest Classifier is best suited model for this as per accuracy

# In[98]:


#========================== Thank you ===================================


# In[ ]:




