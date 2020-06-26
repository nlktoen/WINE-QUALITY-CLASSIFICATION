#!/usr/bin/env python
# coding: utf-8

# # RED WINE CLASSIFICATION PROJECT

# # I) DATA INSPECTION

# 1) IMPORT LIBRARIES USED AND THE RED-WINE DATASET

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


# In[89]:


wine_df = pd.read_csv('C:\Kat\Tuyen\Project\winequality-red.csv')


# In[90]:


#Showing the dataframe
wine_df


# 2) SHAPE OF DATA AND GENERAL DESCRIPTIONS

# In[18]:


wine_df.shape


# In[20]:


#(1599,12) means dataset includes 1599 datapoints with 12 features (attributes)
#The last attribute ('quality') is the target column (supervised label)


# In[70]:


#Data general statistical numbers
wine_stats=wine_df.describe().round(decimals=2)
wine_stats


# As we can see from the table, the number of attribute vectors, their mean, standard deviation, minimum/maximum, 1st-2nd-3rd quartiles

# In[51]:


#Data Correlation matrix
cor_matrix=wine_df.corr().round(decimals=1)
cor_matrix


# This matrix is a significant tool to get insights of the correlation between different fields. Values range from -1 to 1, the more the absolute value of them closer to one, the stronger the relationship becomes. Their sign illustrate types of relationship ('-' : negative relationship, '+' : positive relationship)

# For instance, we take the correlation between quality and other attributes, it is evident to see that alcohol has the largest correlation (0.476) and positive relationship with quality. That means the higher the wine alcohol level is, the better it becomes

# In[52]:


#We can visualize this correlation matrix
import seaborn as sns
sns.heatmap(cor_matrix, annot = True)


# In[66]:


#Another visualization of how each of attribute affect others
from pandas.plotting import scatter_matrix
scatter_matrix(wine_df)
plt.show()


# 3) VISUALIZATION OF DATASET

# In[54]:


#Histogram of each attribute
import matplotlib.pyplot as plt
wine_df.hist(bins=50, figsize=(20, 15))
plt.show()


# According to those histograms, we can see attributes: chlorides, density, residual sulphates and target column(quality) have a quite normal distribution

# However, more importantly, the regconition of some attribute has null ('0') values, which is impossible in term of wine indexes and might affect the analysis in future, such as: Acid Citric. So we have to processed those data for better understanding

# # II) DATA CLEANING AND TRANSFORMATION

# 1) DEALING WITH NULL VALUES

# The technique I use here is replacing them with their median value

# Our dealing target is Acid Citric

# In[91]:


# Calculate the median value for Acid Citric
median_ac = wine_df['citric acid'].median()
# Substitute it in the Acid Citric column of the dataset where values are 0
wine_df['citric acid'] = wine_df['citric acid'].replace(
    to_replace=0, value=median_ac)


# We check this acid citric graph again

# In[59]:


wine_df['citric acid'].hist(bins=50, figsize=(20, 15))
plt.show()


# Here all the '0' have been replaced

# 2) DEALING WITH OUTLIERS

# The technique I use here is removing them all from the dataset as they might have negative impact on my classification model

# The below codes follow this logic:

# a) For each column, first it computes the Z-score of each value in the column, relative to the column mean and standard deviation. 

# b) It takes the absolute of Z-score because the direction does not matter, only if it is below the threshold. (here my target is extreme outliers - which are further more than 3 sd from mean value)

# c) All(axis=1) ensures that for each row, all column satisfy the constraint. 

# d) Change the dataframe based on the result

# In[92]:


from scipy import stats
wine_df=wine_df[(np.abs(stats.zscore(wine_df)) < 3).all(axis=1)]


# In[93]:


wine_df


# As the result, there are  148 datapoints that do not meet standards and are removed from the  dataset (9.3% reduction)

# 3) FEATURE SCALING

# Look back to the statistical description of dataframe

# In[94]:


wine_stats=wine_df.describe().round(decimals=2)
wine_stats


# It's obvious that whereas some features have highly larger than '1.0' range as: fixed acidity(7.5), free sulfur dioxide(46)... others are just ranging from 0 to 1. This difference might affect classification depends on distance such as KNeighbor. Moreover, some learning algorithms don't work very well if the features have a different set of values. For this reason we need to apply a proper scaling system.

# The scaling system I choose here is Standardization

# In[101]:


from sklearn.preprocessing import MinMaxScaler as Scaler

scaler = Scaler(feature_range=(-1,1))
scaler.fit(wine_df.iloc[:,:11])
wine_scaled = scaler.transform(wine_df.iloc[:,:11])


# In[109]:


#Scaled values become a 2D array
wine_scaled


# In[111]:


#Return this 2D array back to dataframe, however the 'quality' column is removed
wine_scaled_df = pd.DataFrame(wine_scaled)


# In[112]:


#Add the target column(quality)
wine_scaled_df['11']=wine_df['quality']


# In[113]:


#Return the original names of these columns
wine_scaled_df.columns=wine_df.columns


# In[114]:


#DataFrame showing
wine_scaled_df


# As we can see now all the attributes (except the target column) are standardized (ranging from -1 to 1)

# # III) TESTING MULTIPLE MODELS

# 1) SPLITTING THE DATASET INTO TRAIN AND TEST SET

# In this case, I want to split the it into to train and test set with ratio 0.75 : 0.25, respectively

# In[146]:


X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(wine_scaled,wine_df.quality,test_size=0.25,random_state=5)


# 2) BUILDING AND TESTING MODELS

# Right now, we didnot know which model is the best for our classification, I train and test each of them

# To avoid overfitting, I split the dataset into many different folds for training and testing

# In[141]:


#Import all the learning algorithms we want to test
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# In[142]:


#Import some utilities of sklearn to compare algorithms
from sklearn import  model_selection
from sklearn.metrics import classification_report #Reporting metric
from sklearn.metrics import confusion_matrix #Confusion_matrix Reporting
from sklearn.metrics import accuracy_score #Accuracy calculating


# In[143]:


# Prepare the configuration to run the test
results=[]
names=[]
seed=7


# In[149]:


# Prepare an array with all the algorithms
models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('CARD',DecisionTreeClassifier()))
models.append(('DTR',DecisionTreeRegressor()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
models.append(('LSVC',LinearSVC()))
models.append(('RFC',RandomForestClassifier()))


# In[150]:


#Evaluate each model in turn
for name,model in models:
    kfold=model_selection.KFold(n_splits=10,random_state=seed)
    cv_results=model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg="%s:%f(%f)"%(name,cv_results.mean(),cv_results.std())
    print(msg)


# In[151]:


# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# It looks like that using this comparison method, the most performant algorithm is RFC

# # IV) BUILDING THE BEST MODEL FOR PREDICTION

# 1) FINDING THE BEST PARAMETER FOR RFC

# In[157]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['gini','entropy'],
    'n_estimators':[10,50,100]
}

model_rfc = RandomForestClassifier()

grid_search = GridSearchCV(
    model_rfc, param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, Y_train)


# The parameter above is the best parameter for RandomForestClassifier model, I will use it the build the model

# In[159]:


# Print the bext score found
grid_search.best_score_


# 2) APPLY THE BEST PARAMETERS TO THE MODEL AND TRAIN IT

# In[160]:


# Create an instance of the algorithm using parameters
# from best_estimator_ property
rfc = grid_search.best_estimator_


# In[161]:


# Use the whole dataset to train the model
X = np.append(X_train, X_test, axis=0)
Y = np.append(Y_train, Y_test, axis=0)


# In[163]:


# Train the model
rfc.fit(X, Y)


# In[168]:


wine_df.describe().round(decimals=2)


# # V) MAKE PREDICTIONS

# In[169]:


# We create a new (fake) wine infomation
new_wine = pd.DataFrame([[8.0, 0.6, 0.3, 5, 18, 30, 65,1,3,0.8,12]])
# We scale those values like the others
new_wine_scaled = scaler.transform(new_wine)


# In[170]:


# We predict the outcome
prediction = rfc.predict(new_wine_scaled)


# In[171]:


# A value of "1" means that this person is likley to have type 2 diabetes
prediction


# Prediction points out that this red wine will score 6.0 in quality

# # VI) CONCLUSION

# We finally find a score of 69.1% using RFC algorithm and parameters optimisation. Please note that there may be still space for further analysis and optimisation, for example trying different data transformations or trying algorithms that haven't been tested yet. Once again I want to repeat that training a machine learning model to solve a problem with a specific dataset is a try / fail / improve process.

# # VII) CREDIT

# https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009?fbclid=IwAR3FNcoe7yYcmHHD0fv-Mmk7aEhW2KBPLyzaAkxPlqy_vm3o72HtwJkaV1E

# In[ ]:




