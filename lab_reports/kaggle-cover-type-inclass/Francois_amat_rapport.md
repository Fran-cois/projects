
# Kaggle 
# Cover Type Prediction of Forests
### Author : François Amat
### Kaggle ID : AmatFrançois 
### Kaggle Score : 0.95713
### contact : amat.francois@gmail.com


First, I import all the libraries I need: 


```python
import pandas as pd

import numpy as np

import time 

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier

from sklearn.decomposition import PCA
```

Then, I read and store to a dataframe the train and test set.


```python
df = pd.read_csv("datasets/train-set.csv")
dftest = pd.read_csv("datasets/test-set.csv")
f = open("results_accuracy.out",'w')
```

I split the data into the data I have `df_X` and the data I want to predict `df_y`, I also remove the Id which are not important for the classification.
Then, I create a train and test set in order to get measures to get the best classifier.


```python
IDS = df.Id
df_y = df.Cover_Type
df_X = df.drop(columns=['Id','Cover_Type'])
pca = PCA(n_components=7)
pca.fit_transform(df_X)
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.33, random_state=43)
```

I create a list of classifier I want to test, with adjustable features. 


```python
max_depth, n_estim = 1000, 1000  # best found during iteration on the kaggle judge.
max_depth, n_estim = 5, 5 # In order to quick test.

clf = RandomForestClassifier(n_estimators=n_estim,n_jobs=15,criterion='entropy')
clfBoost = XGBClassifier(nthread=-1,max_depth=max_depth,n_estimators=n_estim)
clf_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 10), random_state=1)
classifiers = [
    ('classifier', clf),
    ('classifierXBGBOOST', clfBoost),
    ('classifierMlp',clf_mlp)
]
```

I create a function to evaluate the clf and store the results in the way that the kaggle system can judge.


```python
def evaluate(_clf,name="out"):
    _clf.fit(df_X, df_y)  # so that parts of the original pipeline are fitted
    print(accuracy_score(_clf.predict(X_test),y_test),file=f)
    pred = _clf.predict(dftest.drop(columns=['Id']))
    print("predict done")
    df_to_csv = pd.DataFrame( data = np.array(dftest.Id), columns=['Id'] )
    df_to_csv['Cover_type'] = pd.DataFrame(data=pred)
    df_to_csv.to_csv(name +'.csv',index=False)
```

Then I fit all classifier, at the end only the xgboost, on all the data. Creating the submit file in the process.


```python
for classfier in classifiers:
    start = time.perf_counter()
    print("===========================================")
    print("new clf:",classfier[0])
    clf_ = classfier[1]
    evaluate(clf_, name=classfier[0])
    print("classfier used:",classfier[0],"accuracy score:", accuracy_score(clf_.predict(X_test),y_test), file=f)
    elapsed = time.perf_counter() - start 
    print('Elapsed %.3f seconds.' % elapsed)
    print("===========================================")
    print("====".replace('=',"\n"))
```

    ===========================================
    new clf: classifier
    predict done
    Elapsed 8.787 seconds.
    ===========================================
    
    
    
    
    
    
    
    
    ===========================================
    new clf: classifierXBGBOOST
    predict done
    Elapsed 74.969 seconds.
    ===========================================
    
    
    
    
    
    
    
    
    ===========================================
    new clf: classifierMlp
    predict done
    Elapsed 293.609 seconds.
    ===========================================
    
    
    
    
    
    
    
    


My final score: 0.95713
I have Changed the max_depth, n_estim , PCA  parameters in order to get the best score. 

However, I think I can still improve this score with more time and with a grid search.
