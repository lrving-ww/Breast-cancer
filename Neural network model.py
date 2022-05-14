# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 16:21:47 2020

@author: Administrator
"""


# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 09:56:28 2020

@author: Administrator
"""
# In[1]
import seaborn as sns
import pandas as pd
import scipy as sp
import numpy as np
import time
seed = 14
np.random.seed(seed)

from keras import backend as K, activations
from keras import layers
from keras import regularizers
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Dense, Activation, Input
from keras.utils import np_utils

from scipy.special import comb 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score

from matplotlib import pyplot as plt
# In[1]
myTrait = 'rs'  # IFNb, LPS, dNS1, unstim
# myTrait = 'gender'      # F, M
# myTrait = 'ethnicity'   # African-American, Caucasian, East Asian, MULTI-RACIAL
features_table = pd.read_csv("H:/ruxianai/rs.csv" )#the data file from Date preprocess
features=features_table

GE_table= pd.read_csv('H:/ruxianai/genes2.txt', sep=',')#the data file from Date preprocess
GE=GE_table


normalize = True
# normalize = False

if normalize:
    GE = (GE - GE.mean())/(GE.std()) # normalizing GE by z-score for each gene
    print("Normalized GE")
else:
    print("Original GE")
    


# In[1]
num_genes = GE.shape[1]
num_samples = GE.shape[0]

print('Number of genes: %i' %num_genes)
print('Number of samples: %i' %num_samples)

# In[]
# Creating the input matrix X and output matrix Y:
X = GE.values.astype(float)  
Y = features[myTrait]

classes, counts = np.unique(Y, return_counts=True)
num_classes = classes.size
print(dict(zip(classes, counts))) # Check the number of samples in each lable

# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encodedY = encoder.transform(Y)

# Convert integers to dummy variables
dummyY = np_utils.to_categorical(encodedY)

# In[]
def build_shallow_network_model(X, dummyY, hidden_dim):
    input_dim = X.shape[1]  # 414 genes measures for each sample
    output_dim = dummyY.shape[1]  # 4 classes in case of stimulations & ethnicity, 2 classes in the case of gender
    
    inputs = Input(shape=(input_dim,))
    hidden = Dense(hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    outputs = Dense(output_dim, activation='softmax')(hidden)
    myModel = Model(inputs=inputs, outputs=outputs)
    myModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    
    return myModel
    
# In[]
def build_two_layer_network_model(X, dummyY, hidden_dim, hidden_dim_2):
    input_dim = X.shape[1]  # 414 genes measured for each sample
    output_dim = dummyY.shape[1]  # 4 classes in case of stimulations & ethnicity, 2 classes in the case of gender
    
    inputs = Input(shape=(input_dim,)) 
    hidden1 = Dense(hidden_dim, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
    hidden2 = Dense(hidden_dim_2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(hidden1)
    predictions = Dense(output_dim, activation='softmax')(hidden2)
    myModel = Model(inputs=inputs, outputs=predictions)
    myModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return myModel

# In[]
def build_logistic_regression_model(X, dummyY):
    input_dim = X.shape[1]  # 414 genes measures for each sample
    output_dim = dummyY.shape[1]  # 4 classes in case of stimulations & ethnicity, 2 classes in the case of gender
    
    inputs = Input(shape=(input_dim,))
    outputs = Dense(output_dim, activation='softmax')(inputs)
    
    myModel = Model(inputs=inputs, outputs=outputs)
    myModel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])   
    
    return myModel

    
# In[]
from sklearn.metrics import precision_score,recall_score,roc_auc_score,roc_curve,auc
from decimal import Decimal
def cross_validation_prediction(X, encodedY, dummyY, seed, hidden_dim, hidden_dim_2, debug):
    # Create 10-fold cross validation split:
    k = 10

    cv_scores = pd.DataFrame(np.zeros((k, 5)), columns = ['accuracy','precision','recall','f1','auc'])
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    i=0
 
    
    #for _ in range(n_repeats):
    
    
    for train, test in kfold.split(X,encodedY):
        # Choose the neural network architecture of the model:
        myModel = None
        if hidden_dim > 0:
            if hidden_dim_2 == 0:
                if i==0:
                    print("shallow neural network with " + str(hidden_dim) + " dim hidden layer:")
                myModel = build_shallow_network_model(X, dummyY, hidden_dim)  
            else:
                if i==0:
                    print("two-layer neural network with " + str(hidden_dim) + " and " + str(hidden_dim_2) + " dim hidden layer:")
                myModel = build_two_layer_network_model(X, dummyY, hidden_dim, hidden_dim_2)
        else:
            if i==0:
                print("logistic regression model:")
            myModel = build_logistic_regression_model(X, dummyY)

        # Fit model according to train set:    
        print("   split %i" %i)
        start_time = time.time()
        myModel.fit(X[train], dummyY[train], epochs=10, batch_size=5, verbose=0)
        cv_scores['time'] = time.time() - start_time

        # Evaluate accuracy on test set:
        scores = myModel.evaluate(X[test], dummyY[test], verbose=0)
        cv_scores['accuracy'][i] = scores[1] 
        if debug==True:
            print("%s: %.2f%%" % (myModel.metrics_names[1], scores[1]*100))
        
        # Eveluate precision, recall and f1 on test set:
        probs = myModel.predict(X[test], batch_size=5, verbose=0)
        predicted = np.argmax(probs, axis=1)
        true = encodedY[test]

        cv_scores['precision'][i] = precision_score(true, predicted, average='weighted')
        cv_scores['recall'][i] = recall_score(true, predicted, average='weighted')
        cv_scores['f1'][i] = f1_score(true, predicted, average='weighted')
        #cv_scores['auc'][i] = roc_auc_score(true, predicted, average='weighted')
        fpr,tpr,threshold = roc_curve(dummyY[test].ravel(), probs.ravel()) 
        roc_auc = auc(fpr,tpr)
        cv_scores['auc'][i] = roc_auc
        
        i = i+1

        #################
        if debug == True:
            classes, counts = np.unique(predicted, return_counts=True)
            print("predicted:" + str(dict(zip(classes, counts))))
            classes, counts = np.unique(true, return_counts=True)
            print("true:" + str(dict(zip(classes, counts))))
            print("")
   

        
        
    from pylab import mpl  
    mpl.rcParams['font.sans-serif'] = ['SimHei'] 
    #compute ROC curve
    fpr,tpr,threshold = roc_curve(dummyY[test].ravel(), probs.ravel()) 
    roc_auc = auc(fpr,tpr) 
    plt.figure(0).clf()
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='blue',
     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig("F:/ROC curve/%d.jpg"%(j+1))
    plt.show()
    return cv_scores  
   

   

    
# In[]
import warnings
warnings.filterwarnings("ignore")
#architectures = [[4,0],[8,0],[8,2],[16,8]]
architectures = [[8,0],[16,0],[32,0],[64,0],[128,0],[8,2],[16,8],[32,16],[64,32],[128,64],[0,0]]
#architectures = [[8,0]]
mean_results = pd.DataFrame(np.zeros((len(architectures), 5)), columns = ['accuracy','precision','recall','f1','auc'])
std_results = pd.DataFrame(np.zeros((len(architectures), 5)), columns = ['accuracy','precision','recall','f1','auc'])
#S1=pd.DataFrame(np.zeros((10, 5)), columns = ['accuracy','precision','recall','f1','auc'])
j=0

print("Testing " + str(len(architectures)) + " architectures for " + myTrait + ":")
print("----------------------------------------")
for architecture in architectures: 
    cvscores = cross_validation_prediction(X, encodedY, dummyY, seed, architecture[0], architecture[1], debug=False)
    mean_results.iloc[j,:] = cvscores.mean()
    std_results.iloc[j,:]=cvscores.std()
    j=j+1
    
    
mean_results=round(mean_results,5)
mean_results
f = open (r'F:/Model performance.txt','w')
print (mean_results,file = f)
f.close()
mean_results
# In[]mean_results

mean_results_copy = mean_results.copy()
new_results = pd.DataFrame(index=mean_results.index,columns = mean_results.columns)
for row in range(mean_results.shape[0]):
    for col in range(mean_results.shape[1]-1):
        new_results.iloc[row,col] = ("%.4f (+/- %.4f)" % (mean_results.iloc[row,col], std_results.iloc[row,col]))
    new_results.iloc[row,col+1] = ("%.4f" % mean_results.iloc[row,col+1])
    
new_results

new_results.to_excel('F:/The performance of 10 model.xlsx', myTrait)

# In[]The roc curce of the optimal neural network

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

seed=14#model = DecisionTreeClassifier(max_depth=9,random_state=0)
k=10
cv_scores = pd.DataFrame(np.zeros((k, 5)), columns = ['accuracy','precision','recall','f1','auc'])
mean_results= pd.DataFrame(np.zeros((k, 5)), columns = ['accuracy','precision','recall','f1','auc'])
#mean_results = pd.DataFrame(np.zeros(k, 5), columns = ['accuracy','precision','recall','f1','auc'])
i=0
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)


for train,test in kfold.split(X,encodedY):
    myModel = None
    myModel = build_shallow_network_model(X, dummyY, 8) 
    myModel.fit(X[train], dummyY[train], epochs=10, batch_size=5, verbose=0)
    scores = myModel.evaluate(X[test], dummyY[test], verbose=0)
    cv_scores['accuracy'][i] = scores[1]
    
    
    probs = myModel.predict(X[test], batch_size=5, verbose=0)
    predicted = np.argmax(probs, axis=1)
    true = encodedY[test]
    
    cv_scores['precision'][i] = precision_score(true, predicted, average='weighted')
    cv_scores['recall'][i] = recall_score(true, predicted, average='weighted')
    cv_scores['f1'][i] = f1_score(true, predicted, average='weighted')
    
    fpr,tpr,threshold = roc_curve(dummyY[test].ravel(), probs.ravel()) 
    roc_auc = auc(fpr,tpr)
    cv_scores['auc'][i] = roc_auc
    
    #########draw roc curve 
    from pylab import mpl  
    mpl.rcParams['font.sans-serif'] = ['SimHei']     
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='blue',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title('the optimalneural network roc curve in Validation set',fontsize=20)
    plt.tight_layout(pad=1.08)
    plt.legend(loc="lower right",fontsize=20)
    plt.savefig("F:/the optimal neural network roc curve/%d.jpg"%(i+1))
    plt.show()  
    i = i+1
    
mean_results=round(cv_scores.mean(),5)    
mean_results








# In[]The accuracy of the optimal neural network model is verified by 10 fold cross


from sklearn.datasets import make_blobs
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from numpy import mean
from numpy import std
from matplotlib import pyplot
from pandas.core.frame import DataFrame


 
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#i=0
n_repeats = 10
scores2=list()
scores3=list()
std_cha=list()
#df = pd.DataFrame() #
for _ in range(n_repeats):
    for train, test in kfold.split(X,encodedY):
        myModel = None
        myModel = build_shallow_network_model(X, dummyY, 8) 
        myModel.fit(X[train], dummyY[train], epochs=10, batch_size=5, verbose=0)
        _,scores1 = myModel.evaluate(X[test], dummyY[test], verbose=0)
        print('> %.3f' % scores1)
        scores2.append(scores1)
    std_cha.append(std(scores2))
    scores3.append(np.mean(scores2))
        #i=i+1


# In[]
#In[]The accuracy bar chart of the optimal model was drawn
import matplotlib.pyplot as plt
import numpy as np
 

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


group=['1','2','3','4','5','6','7','8','9','10'] 

scores3
std_cha
bar_width = 0.4 
index_male = np.arange(len(group)) 
index_female = index_male + bar_width 
plt.figure(figsize=(12,6))

s=12
tup1=plt.bar(index_male, height=scores3, width=bar_width, color='mediumpurple', label='准确率')
def autolabel(tup1):
  for rect in tup1:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, '%s' % round(height,3),verticalalignment="center",
	horizontalalignment="center",fontsize=s)
autolabel(tup1)


tup2=plt.bar(index_female, height=std_cha, width=bar_width, color='skyblue', label='标准差')
def autolabel(tup2):
  for rect in tup2:
    height = rect.get_height()
    plt.text(rect.get_x()+rect.get_width()/2., 2*height, '%s' % round(height,3),verticalalignment="center",
	horizontalalignment="center",fontsize=s)
autolabel(tup2)


plt.legend() 

plt.xticks(index_male + bar_width/2, group,fontsize=s)

plt.yticks(np.arange(0.0,1.2,0.1),fontsize=s) 
plt.xlabel('times of 10 fold cross validation',fontsize=s) 
plt.ylabel('Accuracy and standard deviation',fontsize=s) 
plt.savefig('F:/p.jpg',dpi=1200)
plt.show()