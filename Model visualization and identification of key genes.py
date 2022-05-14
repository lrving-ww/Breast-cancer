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


# In[]start Model visualization and identification of key genes
myModel=None
myModel=build_shallow_network_model(X,dummyY,8)
kfold=StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
for train,test in kfold.split(X,encodedY):
    print('train,test'% train,test)
    continue

myModel.fit(X[train],dummyY[train],epochs=10,batch_size=5,verbose=0)
scores=myModel.evaluate(X[test],dummyY[test],verbose=0)
scores[1]



# In[]Significance detection
import os
from vis.utils import utils

myModel.layers[-1].activation = activations.linear
os.makedirs('/tmp/', exist_ok=True)
myModel = utils.apply_modifications(myModel)


# In[]

def my_saliency_map(myModel, class_idx, sample_vec):    
    input_vec = myModel.input
    output_vec = myModel.output[:, class_idx]    
    
    # Loss is defined according to the class we want to maximize:
    loss = K.mean(myModel.output[:, class_idx])  
 
    # Compute the gradient of the output vector w.r.t the input vector + normalize to avoid very small/large gradients:
    grads = K.gradients(output_vec, input_vec)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # Calculate the loss and grads given the input vector:
    iterate = K.function([input_vec], [loss,grads])

    # Gradient descent on the given sample_vec:
    alpha = 0.1
    num_steps = 1
    
    sample_vec = sample_vec.reshape(1,num_genes)
    for i in range(num_steps):
        loss_value, grads_value = iterate([sample_vec])
        grads_value -= grads_value*alpha
    
    return grads_value[0]

    


  # In[]  
  
# Create dataframes of gene expression and saliency maps of size num_genes X num_samples:
genes = GE.columns.values.tolist()
expression_all_samples =  pd.DataFrame(genes, columns=['Gene'])
saliency_all_samples =  pd.DataFrame(genes, columns=['Gene'])

# Create a dataframe of averaged saliency maps of size num_genes X num_classes:
saliency_average_samples = pd.DataFrame(genes, columns=['Gene'])

# Saving the attention scores to an output file:
filepath = 'F:/All significance.xlsx'
writer = pd.ExcelWriter(filepath)

# For every class - 
# (i)  Go through each relevant sample to retrieve gene expression and calculate saliency map 
# (ii) Caclulate the averaged saliency across all relevant samples
class_idx=0
for class_label in np.unique(Y): 
    print(class_label)
 
    i=0
    class_samples_indices = np.where(encodedY[test]==class_idx)[0]
    np.random.shuffle(class_samples_indices)   
    num_class_samples = len(class_samples_indices)
    
    class_expression = pd.DataFrame(np.zeros((num_genes, num_class_samples)),columns=[str(class_label) + '_' + str(x) for x in range(num_class_samples)])
    class_saliency = pd.DataFrame(np.zeros((num_genes, num_class_samples)),columns=[str(class_label)+ '_' + str(x) for x in range(num_class_samples)])
    
    for sample_idx in class_samples_indices:       
        sample_vec = X[test][sample_idx]
        class_expression.iloc[:,i] = sample_vec
        
        saliency_vec = my_saliency_map(myModel, class_idx, sample_vec)
        class_saliency.iloc[:,i] = saliency_vec
        i+=1
    
    expression_all_samples = expression_all_samples.join(class_expression)
    saliency_all_samples = saliency_all_samples.join(class_saliency)
    saliency_average_samples[str(class_label)] = np.mean(class_saliency,axis=1)

    class_saliency = saliency_all_samples['Gene'].to_frame().join(class_saliency)
    class_saliency.to_excel(writer,str(class_label))
    writer.save()   
    class_idx+=1
    
  
 # In[]
from pylab import mpl 
mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


sns.set(font_scale=0.4)

#f,(ax1,ax2) = plt.subplots(2,1, figsize=(1.5,2.25), sharex=True)
f,(ax1,ax2) = plt.subplots(1,2, figsize=(6,3), sharex=True)
g1 = sns.heatmap(expression_all_samples.iloc[:,1:],ax=ax1, cmap="rainbow",vmin=-1,vmax=1)
ax1.set_title('Expression')
ax1.set_ylabel('Genes')
g2 = sns.heatmap(saliency_all_samples.iloc[:,1:],ax=ax2, cmap="rainbow",vmin=-1,vmax=1)
ax2.set_title('Saliency')
plt.savefig('F:/Expression diagram and mapping diagram.jpg', dpi = 1200) 
plt.show()





  
# In[]


def my_activation_maximization(myModel, class_idx):    
    input_vec = myModel.input
    output_vec = myModel.output[:, class_idx] 
    
    # Loss is defined according to the class we want to maximize:
    loss = K.mean(myModel.output[:, class_idx]) 
        
    # Compute the gradient of the loss w.r.t the input vector + normalize to avoid very small/large gradients:
    grads = K.gradients(loss, input_vec)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # Calculate the loss and grads given the input vector:
    iterate = K.function([input_vec], [loss,grads])

    # Start gradient ascent from a random input vector:
    rand_vec = np.random.random((1, input_vec.shape[1]))
    
    # Gradient ascent:
    alpha = 0.1
    num_steps = 5
    for i in range(num_steps):
        loss_value, grads_value = iterate([rand_vec])
        rand_vec += grads_value*1.
        
    rand_vec = np.transpose(rand_vec)
    
    return rand_vec

    
# In[]

# Create a dataframe of activation maximization scores of size num_genes X num_classes:
genes = GE.columns.values.tolist()
act_max =  pd.DataFrame(genes, columns=['Gene'])

# Computing activation maximization for each class:
class_idx=0
for class_label in np.unique(Y): 
    print(class_label)
    
    act_max[str(class_label)] = my_activation_maximization(myModel, class_idx)        
    class_idx+=1
    
filepath = 'F:/Maximum activation value.xlsx'
writer = pd.ExcelWriter(filepath)
act_max.to_excel(writer,'activation_maximization')

filepath = 'F:/Mean significance value.xlsx'
writer = pd.ExcelWriter(filepath)
saliency_average_samples.to_excel(writer,'saliency_average_samples')
saliency_average_samples
writer.save()


# In[]

#sns.set(font_scale=0.2,font='STSong')
from pylab import mpl
import matplotlib.pyplot as plt

sns.set(font_scale=0.2,font='SimHei')
mpl.rcParams['axes.unicode_minus']=False

fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(2.2,1), sharey=True)

sns.heatmap(np.transpose(saliency_average_samples.iloc[:,1:]),cmap="BuPu", ax=ax0, vmin=-1, vmax=1, yticklabels=True)
ax0.set_xlabel('Genes')

ax0.set_title('Mean significance plot')
sns.heatmap(np.transpose(act_max.iloc[:,1:]),cmap="BuPu", ax=ax1, vmin=-6, vmax=6, yticklabels=True)
ax1.set_xlabel('Genes')
ax1.set_title('Activation maximization plot')

plt.subplots_adjust(wspace = 0.3,bottom=0.3)
plt.savefig('F:Mean significance plot and activation maximization plot.jpg', dpi = 1200)





# In[]
print("Correlation between activation maximization and averaged saliency maps:")
act_max.corrwith(saliency_average_samples)






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
    #for i in range(len(architectures)):

    plt.show()
    return cv_scores  




# In[]Use activation maximization to get the first 10 genes
k=15
act_max1=act_max.drop(['Gene'],axis=1)
act_max2=act_max1.columns.values.tolist()
act_max3=np.array(act_max2)
#act_max3=pd.DataFrame(act_max3)
#act_max=act_max.rename(columns={False:'False',True:'True'},inplace=True)
genes = GE.columns.values.tolist()
act_max_scores = act_max.copy()
act_max_scores['avg_abs_activation'] = (act_max[act_max3]).abs().mean(axis=1)
act_max_scores = act_max_scores.sort_values(by=['avg_abs_activation'], ascending=False)
sorted_genes_act_max = act_max_scores['Gene'].tolist()

act_max_scores.head(k)
(act_max_scores.head(k)).to_excel('F:/Activation maximization and absolute value averaging (top 10 in descending order).xlsx')
max=pd.DataFrame(act_max_scores.head(k)['Gene'])
max
max.to_csv('F:/max.csv',index=0)




# In[]The first 10 genes were used to verify the accuracy of the model

# k = 1
# k = 2
# k = 3
# k = 5
k = 10
# k = 20
# k = 50
# k = 100
# k = 200
#Top-K genes based on activaiton mazimization:
top_k_act_max_GE = GE[sorted_genes_act_max[0:k]]
X_top_act_max = top_k_act_max_GE.values.astype(float)  

act_max_cvscores = cross_validation_prediction(X_top_act_max, encodedY, dummyY, seed, 8, 0, debug=False)
print("\nClassification based on top %i activation maximization genes: %.2f%% (+/- %.2f%%)" % (k, np.mean(act_max_cvscores['accuracy']), np.std(act_max_cvscores['accuracy'])))





# In[]The first 10 genes obtained by the significance test

act_max1=saliency_average_samples.drop(['Gene'],axis=1)
act_max2=act_max1.columns.values.tolist()
act_max3=np.array(act_max2)
#act_max3=pd.DataFrame(act_max3)
#act_max=act_max.rename(columns={False:'False',True:'True'},inplace=True)
genes = GE.columns.values.tolist()
saliency_average_samples_scores = saliency_average_samples.copy()
saliency_average_samples_scores['avg_abs_activation'] = (saliency_average_samples[act_max3]).abs().mean(axis=1)
saliency_average_samples_scores = saliency_average_samples_scores.sort_values(by=['avg_abs_activation'], ascending=False)
sorted_genes_saliency_average =saliency_average_samples_scores['Gene'].tolist()

saliency_average_samples_scores.head(15)

saliency_average_samples_scores.to_excel('F:/Average significance and absolute value average (in descending order).xlsx')
(saliency_average_samples_scores.head(15)).to_excel('F:/Average significance and absolute value average (in descending order).xlsx')


average=pd.DataFrame(saliency_average_samples_scores.head(15))['Gene']
average
average.to_csv('F:/average.csv',index=0)


# In[]Model accuracy of the top 10 genes based on significance tests

# k = 1
# k = 2
# k = 3
# k = 5
k = 15
# k = 20
# k = 50
# k = 100
# k = 200
#Top-K genes based on activaiton mazimization:
top_k_saliency_average_GE = GE[sorted_genes_saliency_average[0:k]]
X_top_saliency_average = top_k_saliency_average_GE.values.astype(float)  

saliency_average_cvscores = cross_validation_prediction(X_top_saliency_average, encodedY, dummyY, seed, 8, 0, debug=False)
print("\nClassification based on top %i activation maximization genes: %.2f%% (+/- %.2f%%)" % (k, np.mean(saliency_average_cvscores['accuracy']), np.std(saliency_average_cvscores['accuracy'])))




# In[]

myModel = None
myModel = build_shallow_network_model(X_top_saliency_average, dummyY, 8)  
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
for train, test in kfold.split(X,encodedY):
    continue

myModel.fit(X_top_saliency_average[train], dummyY[train], epochs=10, batch_size=5, verbose=0)





scores = myModel.evaluate(X_top_saliency_average[test], dummyY[test], verbose=0)

print("Model accuracy: " + str(scores[1]*100))
print("Model loss: " + str(scores[0]*100))
myModel.metrics_names######








# In[]The second category

from pylab import mpl
import matplotlib.pyplot as plt

                   
fpr,tpr,threshold = roc_curve(dummyY[test].ravel(), y_score.ravel())
roc_auc = auc(fpr,tpr) 
 

right_index=(tpr+(1-fpr)-1)
right_index=right_index.tolist()
yuzhi=max(right_index)
index=right_index.index(yuzhi)
tpr=tpr.tolist()
tpr_val=tpr[index]
fpr_val=fpr[index]


plt.subplots(figsize=(7.5,5));
lw = 2

plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
a=12
plt.xlabel('False Positive Rate',fontsize=a)
plt.ylabel('True Positive Rate',fontsize=a)
plt.title('ROC Curve',fontsize=a)
plt.legend(loc="lower right",fontsize=a)
plt.savefig('F:/ROC curves of 10 genes corresponding to the model.jpg',dpi=1200)
plt.show()








# In[]Logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X, encodedY)

genes = GE.columns.values.tolist()
beta_df =  pd.DataFrame(genes, columns=['Gene'])
beta_df = beta_df.join(pd.DataFrame(clf.coef_.transpose()))

beta_df['max_abs_coef'] = (beta_df.iloc[:, 1:]).abs().max(axis=1)

filepath = 'F:/the score of Logistic regression.xlsx'
writer = pd.ExcelWriter(filepath)
beta_df.to_excel(writer,'beta')
writer.save()

beta_df = beta_df.sort_values(by=['max_abs_coef'], ascending=False)
sorted_genes_standard_form = beta_df['Gene'].tolist()

(beta_df.head(10)).to_excel('F:/the score of Logistic regression（top 10）.xlsx')




# In[]top 1o genes based on Logistic regression
# Top-K genes based on standard logistic regression:
standard_form_GE = GE[sorted_genes_standard_form[0:k]]
X_standard_form = standard_form_GE.values.astype(float)  

standard_form_cvscores = cross_validation_prediction(X_standard_form, encodedY, dummyY, seed, 8, 0, debug=False)
print("\nClassification based on top %i closed-form genes: %.2f%% (+/- %.2f%%)" % (k, np.mean(standard_form_cvscores['accuracy']), np.std(standard_form_cvscores['accuracy'])))



# In[]
sorted_genes_act_max

# In[]
sorted_genes_standard_form