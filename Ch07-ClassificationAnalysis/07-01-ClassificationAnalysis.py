#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/roitraining/PythonML/blob/Development/Ch07-ClassificationAnalysis/07-01-ClassificationAnalysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# ## Combine the multiple files into one big CSV since we could not load a large file to GitHub.

# In[ ]:


get_ipython().system(' ./combine.sh')


# ### Read in a set of data and examine it

# In[1]:


import pandas as pd
df = pd.read_csv('CreditCardFraud.csv')

print (df.shape, df.columns)
train_size = .3
test_size = .1

print (df.head())
print (df.isFraud.value_counts())
print (df.type.value_counts())


# ### Keep the columns we want and change the type to code numbers instead

# In[2]:


columns = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'isFraud']
df = df[columns]
df.type = pd.Categorical(df.type).codes
print (df.shape, df.columns)
print (df.head())


# ### Prepare train & test sets with desired columns

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
dfNB = df
trainNB_X, testNB_X, trainNB_Y, testNB_Y = train_test_split(dfNB[dfNB.columns[:-1]], dfNB.isFraud, train_size = train_size, test_size = test_size)
print (testNB_Y.value_counts())
print(trainNB_Y.value_counts()/trainNB_Y.count())
print(testNB_Y.value_counts()/testNB_Y.count())
print (trainNB_X[:10])


# ## Create a Naive Bayes model

# In[4]:


from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(trainNB_X, trainNB_Y)


# ### Examine the results of Naive Bayes

# In[8]:


predNB_Y = modelNB.predict(testNB_X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testNB_Y, predNB_Y)
print (cm)

# helper function to print confusion matrix as percentages
def cm_percent(cm, length, legend = True):
    import numpy as np
    if legend:
       print (' PC', 'FP\n', 'FN', 'PW')
    return np.ndarray(shape = (2,2), buffer = np.array([100 *(cm[0][0] + cm[1][1])/length,
       100 * cm[0][1]/length, 100 * cm[1][0]/length, 100 * (cm[1][0] + cm[0][1])/length]))

cmp = cm_percent(cm, len(testNB_Y))
print (cmp)
print (testNB_Y.value_counts())
print (len(testNB_Y))


# ## Save a trained model

# In[9]:


from joblib import dump, load
dump(modelNB, 'modelNB.joblib') 


# ## Load a saved model

# In[10]:


modelNB2 = load('modelNB.joblib')
predNB_Y = modelNB2.predict(testNB_X)
cm = confusion_matrix(testNB_Y, predNB_Y)
print (cm)
cmp = cm_percent(cm, len(testNB_Y))
print (cmp)


# ## Train the Decision Tree model

# In[11]:


from sklearn.tree import DecisionTreeClassifier
dfDT = df
trainDT_X, testDT_X, trainDT_Y, testDT_Y = train_test_split(dfDT[dfDT.columns[:-1]], dfDT.isFraud, train_size = train_size, test_size = test_size)

modelDT = DecisionTreeClassifier()
modelDT.fit(trainDT_X, trainDT_Y)


# ## Examine the results of the Decision Tree

# In[13]:


def important_features(model, columns):
    return pd.DataFrame(model.feature_importances_, columns=['Importance'], index = columns).sort_values(['Importance'], ascending = False)
 
predDT_Y = modelDT.predict(testDT_X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testDT_Y, predDT_Y)
print (cm)
print (cm_percent(cm, len(testDT_Y)))
print (testDT_Y.value_counts(), len(testDT_Y))
print (important_features(modelDT, trainDT_X.columns))


# ## Prepare the data
# ### Logistic Regression requires categorical data be dummy encoded

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
def dummy_code(data, columns, drop_first = True):
    for c in columns:
        dummies = pd.get_dummies(data[c], prefix = c, drop_first = drop_first)
        i = list(data.columns).index(c)
        data = pd.concat([data.iloc[:,:i], dummies, data.iloc[:,i+1:]], axis = 1)
    return data

dfLR = dummy_code(df, ['type'], drop_first = True)
trainLR_X, testLR_X, trainLR_Y, testLR_Y = train_test_split(dfLR.iloc[:,dfLR.columns != 'isFraud'], dfLR.isFraud, train_size = train_size, test_size = test_size)

print (testLR_X.columns)
print (testLR_X.head())


# ## Create a Logistic Regression model

# In[15]:


from sklearn.linear_model import LogisticRegression
modelLR = LogisticRegression(multi_class='auto', solver='lbfgs')
modelLR.fit(trainLR_X, trainLR_Y)
print(modelLR.coef_)


# ## Examine the results of Logistic Regression

# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
predLR_Y = modelLR.predict(testLR_X)
from sklearn.metrics import confusion_matrix
score = modelLR.score(testLR_X, testLR_Y)
mse = np.mean((predLR_Y - testLR_Y)**2)
print (score, mse)

cm = confusion_matrix(testLR_Y, predLR_Y)
print (cm)
cmp = cm_percent(cm, len(testLR_Y))
print (cmp)

predLR_Y1 = modelLR.predict_proba(testLR_X)

from sklearn.metrics import roc_auc_score, roc_curve
roc = roc_auc_score(testLR_Y, predLR_Y)
fpr, tpr, x = roc_curve(testLR_Y, predLR_Y1[:,1])

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label = 'AUC = ' + str(roc))
plt.legend(loc=4)
plt.show()

#import scikitplot.metrics as skplt
#import matplotlib.pyplot as plt
#skplt.plot_roc(testY, predY1)
#plt.show()


# ## Try Logistic Regression with different probability thresholds to change ratio of false negatives and positives

# In[24]:


predLR_Y = modelLR.predict_proba(testLR_X)
print (predLR_Y[:10])
print ('Score', modelLR.score(testLR_X, testLR_Y))

for threshold in range(30, 91, 10):
    predLR_Y1 = np.where(predLR_Y[:,0] >= threshold/100, 0, 1)
    mse = np.mean((predLR_Y1 - testLR_Y)**2)
    cm = confusion_matrix(testLR_Y, predLR_Y1)
    print ('\nTHRESHOLD', threshold, 'MSE', mse)
    print (cm)
    print (cm_percent(cm, len(testLR_Y), legend = False))


# ## Prepare the data for a Neural Network
# ### This time you should not drop the first column when dummy encoding. Additionally, data works better if it is rescaled.

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
# rescale the data
dfNN = dummy_code(df, ['type'], drop_first = False)
print (dfNN.columns)
dfNN[['amount',  'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']] /= dfNN[['amount',  'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']].max()
trainNN_X, testNN_X, trainNN_Y, testNN_Y = train_test_split(dfNN.iloc[:,dfNN.columns != 'isFraud'], dfNN.isFraud, train_size = train_size, test_size = test_size)


# ## Create a Neural Network model

# In[26]:


from sklearn.neural_network import MLPClassifier
modelNN = MLPClassifier(hidden_layer_sizes = (5, 3, 2), activation = 'logistic')
modelNN.fit(trainNN_X, trainNN_Y)


# ## Examine the results of Neural Network predictions

# In[27]:


predNN_Y = modelNN.predict(testNN_X)
cm = confusion_matrix(testNN_Y, predNN_Y)
print (cm)
cmp = cm_percent(cm, len(testNN_Y))
print (cmp)


# ## Create a SVM model

# In[28]:


from sklearn import svm
train_size = .03
test_size = .01
dfSVM = dfNN
trainSVM_X, testSVM_X, trainSVM_Y, testSVM_Y = train_test_split(dfSVM.iloc[:,dfSVM.columns != 'isFraud'], dfSVM.isFraud, train_size = train_size, test_size = test_size)

def do_SVM(kernel, gamma):
    print ("\nKernel:", kernel, "Gamma:", gamma)
    modelSVM = svm.SVC(gamma = gamma,  kernel = kernel)
    modelSVM.fit(trainSVM_X, trainSVM_Y)
    print (modelSVM.score(testSVM_X, testSVM_Y))

    predSVM_Y = modelSVM.predict(testSVM_X)
    cm = confusion_matrix(testSVM_Y, predSVM_Y)
    print (cm)

do_SVM('linear', gamma='auto')

for kernel in ['rbf', 'poly', 'sigmoid']:
    for gamma in ['auto', 10, 100]:
        if not (kernel == 'poly' and gamma == 100):
           do_SVM(kernel, gamma)


# In[43]:


modelSVM = svm.SVC(gamma = 100)
modelSVM.fit(trainSVM_X, trainSVM_Y)
print(modelSVM.score(testSVM_X, testSVM_Y))
predSVM_Y = modelSVM.predict(testSVM_X)
print(confusion_matrix(testSVM_Y, predSVM_Y ))


# ## Ensemble Learning

# ## Create and train a Random Forest Classifier

# In[33]:


from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier(n_estimators=10)
trainRF_X, trainRF_Y, testRF_X, testRF_Y = trainDT_X, trainDT_Y, testDT_X, testDT_Y
modelRF.fit(trainRF_X, trainRF_Y)


# ## Test the accuracy of the predictions and examine important features

# In[34]:


predRF_Y = modelRF.predict(testRF_X)
from sklearn import metrics
print ("Accuracy:",metrics.accuracy_score(testRF_Y, predRF_Y))
cm = confusion_matrix(testRF_Y, predRF_Y)
print (cm)

import pandas as pd
feature_imp = pd.Series(modelRF.feature_importances_,index=trainRF_X.columns).sort_values(ascending=False)
print (feature_imp)


# ## Visualize important features

# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# ## Try removing less important features and retrain it

# In[36]:


newTrainRF_X = trainRF_X[['newbalanceDest', 'oldbalanceOrg', 'amount', 'oldbalanceDest']]
newTestRF_X = testRF_X[['newbalanceDest', 'oldbalanceOrg', 'amount', 'oldbalanceDest']]
from sklearn.ensemble import RandomForestClassifier
modelRF = RandomForestClassifier(n_estimators=10)
modelRF.fit(newTrainRF_X, trainRF_Y)


# ### In this case the accuracy did not go up, but in many cases it does

# In[37]:


newpredRF_Y = modelRF.predict(newTestRF_X)
from sklearn import metrics
print ("Accuracy:",metrics.accuracy_score(testRF_Y, newpredRF_Y))
cm = confusion_matrix(testRF_Y, newpredRF_Y)
print (cm)

import pandas as pd
feature_imp = pd.Series(modelRF.feature_importances_,index=newTrainRF_X.columns).sort_values(ascending=False)
print (feature_imp)


# In[40]:


from sklearn.ensemble import VotingClassifier
modelVC = VotingClassifier(estimators=[('dt', modelDT), ('nb', modelNB)], voting='hard')
modelVC.fit(trainDT_X, trainDT_Y)


# In[41]:


print(modelVC.score(testDT_X,testDT_Y))

predVC_Y = modelVC.predict(testDT_X)
cm = confusion_matrix(testDT_Y, predVC_Y)
print (cm)


# In[68]:


from sklearn.neural_network import MLPClassifier
modelNN = MLPClassifier(hidden_layer_sizes = (5, 3, 2), activation = 'logistic')
modelNN.fit(trainLR_X, trainLR_Y)

predLR = (modelLR.predict_proba(testLR_X))[:,0]
predNN = (modelNN.predict_proba(testLR_X))[:,0]

predAvg = (predLR + predNN) / 2
predAvg1 = np.where(predAvg >= .7, 0, 1)

print (confusion_matrix(testLR_Y, predAvg1))


# # End of notebook
