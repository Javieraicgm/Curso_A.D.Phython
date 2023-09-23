import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import matthews_corrcoef,cohen_kappa_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,stratify=y,random_state=1)

pipe_svc = make_pipeline(StandardScaler(),SVC(random_state=1))

param_range = [0.0001*0.5,0.0001,0.001*0.5,0.001,0.01*0.5,0.01,0.1*0.5,0.1,
               1.*0.5,1.,10.*0.5,10.,100.*0.5,100.,1000.*0.5,1000.]

param_grid = [{'svc__C': param_range,'svc__kernel': ['linear']},
              {'svc__C': param_range,'svc__gamma': param_range,'svc__kernel': ['rbf']},
              {'svc__C': param_range,'svc__gamma': param_range,'svc__kernel': ['poly']},
              {'svc__C': param_range,'svc__gamma': param_range,'svc__kernel': ['sigmoid']}]

gs = GridSearchCV(estimator=pipe_svc,param_grid=param_grid,scoring='accuracy',cv=10,n_jobs=1)
gs = gs.fit(X_train, y_train)

clf = gs.best_estimator_

print('Los mejores Hiperparámetros son: \n',gs.best_params_)

y_pred = clf.predict(X_test)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Matriz Confusión: \n', confmat)
print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
print('MCC: %.3f' % matthews_corrcoef(y_true=y_test, y_pred=y_pred))
print('Kappa: %.3f' % cohen_kappa_score(y1=y_test, y2=y_pred))
