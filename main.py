
# Load libraries
#import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
#from matplotlib import pyplot
#from pandas import read_csv
#from pandas.plotting import scatter_matrix
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

# Load dataset
data = r'C:\Users\Gustavo Valentim\OneDrive\Área de Trabalho\python\mi_project\MI.csv'
names = list()
first_day = [92, 93, 94, 99, 100, 101, 102, 103, 104]
for i in range(124):
   names.append(i)
# transforma em data frame
dataset_df = pd.read_csv(data, names=names)
# cópia
data_pre = dataset_df.copy()
# remove colunas dia1/dia3
#data_first = data_pre.drop(columns=first_day)
data_third = data_pre.drop(columns=[0])
names.remove(0)

for i in names:
   # converte a coluna para um tipo númerico
   data_third[i] = pd.to_numeric(data_third[i], errors='coerce')
   # data_third[i] = pd.to_numeric(data_third[i], errors= 'coerce')

for colums in names:
   mediana = data_third[colums].median()
   data_third[colums] = data_third[colums].fillna(mediana)

data_first = data_third.drop(columns=first_day)
array = data_third.values
array_f = data_first.values
X = array[:, :111]
y = array[:, 113]  # the last is 122
X_f = array_f[:, :102]
y_f = array_f[:, 113]  # the last is 113
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.30, random_state=1)
X_ftrain, X_fvalidation, Y_ftrain, Y_fvalidation = train_test_split(
    X_f, y_f, test_size=0.30, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('KNN', KNeighborsClassifier()))

# evaluate each model in turn
results = []
results_f = []
nomes = []
for name, model in models:
   # oito pois a saida sao oito grupos antes estava 10
   kfold = StratifiedKFold(n_splits=8, random_state=1, shuffle=True)
   cv_results = cross_val_score(
       model, X_train, Y_train, cv=kfold, scoring='accuracy')
   cv_results_f = cross_val_score(
       model, X_ftrain, Y_ftrain, cv=kfold, scoring='accuracy')
   results_f.append(cv_results_f)
   results.append(cv_results)
   nomes.append(name)
   print('Primeiro dia  = %s: %f (%f)' %
         (name, cv_results_f.mean(), cv_results_f.std()))
   print('Terceiro dia  = %s: %f (%f)' %
         (name, cv_results.mean(), cv_results.std()))

# Handling Imbalanced Datasets
sm = SMOTEENN(random_state=2)
X_train_sen, Y_train_sen = sm.fit_resample(X_train, Y_train.ravel())
# Make predictions on validation dataset  https://github.com/pik1989/DataBalancingTechniques/blob/main/Sampling_Techniques.ipynb
#model = ()
#model.fit(X_train_sen, Y_train_sen)
#predictions = model.predict(X_validation)
# Evaluate predictions
#f1 = f1_score(Y_validation, predictions)


#print(accuracy_score(Y_validation, predictions))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
# Curva ROC
#logitic_roc_auc = roc_auc_score(Y_validation, predictions)
#fpr, tpr , thresholds = roc_curve(Y_validation,model.predict)

#print('f1_score: ', f1)
