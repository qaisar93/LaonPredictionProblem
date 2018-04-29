
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import cross_val_score


#Reading the datasets

x_train=pd.read_excel('data.xlsx')
y_train=pd.read_excel('label.xlsx')


x_train.describe()
x_train.info()

x_train=x_train.drop(['Loan_ID'],axis=1)
x_train.info()


#labeling the categorical values
le=LabelEncoder()

x_train['Gender']=le.fit_transform(x_train['Gender'])
x_train['Married']=le.fit_transform(x_train['Married'])
x_train['Dependents']=le.fit_transform(x_train['Dependents'])
x_train['Education']=le.fit_transform(x_train['Education'])
x_train['Self_Employed']=le.fit_transform(x_train['Self_Employed'])
x_train['Property_Area']=le.fit_transform(x_train['Property_Area'])

y_train=le.fit_transform(y_train['Target'])



#standardization
sc=StandardScaler()
sc.fit_transform(x_train)


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

#tuning hyper parameters
logisticRegr = LogisticRegression(solver = 'lbfgs')
c_options=[.0001,.001,.01,.1,1,10]
param_grid=dict(C=c_options)
print(param_grid)
grid=GridSearchCV(logisticRegr,param_grid,cv=10,scoring='accuracy')
grid.fit(x_train,y_train)
#grid.grid_scores_
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)
#finished tuning

#tuned classfier
tuned_logisticRegr=LogisticRegression(C=0.1,solver = 'lbfgs')
score_cross_val_l=cross_val_score(tuned_logisticRegr,x_train,y_train,cv=10)

tuned_logisticRegr.fit(x_train, y_train)
predictions = tuned_logisticRegr.predict(x_train)
cm = metrics.confusion_matrix(y_train, predictions)

print("=============================Logistic Regression============================ ")

print("mean cross validation score with standard deviation ")
print(score_cross_val_l.mean(),'+/- ',score_cross_val_l.std())
print("confusion matrix")
print(cm)




#k nearsest neighbors
from sklearn.neighbors import KNeighborsClassifier


#tuning hyperparamter K
knn = KNeighborsClassifier()
k_range=[]
for i in range(1,21):
    k_range.append(i)


param_grid=dict(n_neighbors=k_range)
print(param_grid)

grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')

grid.fit(x_train,y_train)
#grid.grid_scores_
print(grid.best_score_)
print(grid.best_params_)

#finished tuning

#tuned classifier
tuned_knn=KNeighborsClassifier(n_neighbors=7)

score_cross_val_k=cross_val_score(tuned_knn,x_train,y_train,cv=10)

tuned_knn.fit(x_train,y_train)
predict = tuned_knn.predict(x_train)
km= metrics.confusion_matrix(y_train, predict)


print("==================================knn=============================== ")
print("mean cross validation score  with standard deviation")
print(score_cross_val_k.mean(),' +/- ',score_cross_val_k.std())
print("confusion matrix")
print(km)



#support vector classifier
from sklearn.svm import SVC

#tuning hyper parameters C and Gamma
svc=SVC()

c_options=[.0001,.001,.01,.1,1,10]
gamma_options=[.001,.01,.1,1]
kernel_options=['rbf']
param_grid=dict(C=c_options,gamma=gamma_options)
print(param_grid)


grid=GridSearchCV(svc,param_grid,cv=10,scoring='accuracy')
grid.fit(x_train,y_train)
print(grid.best_score_)
print(grid.best_params_)
#print(grid.best_estimator_)

#end tuning

#tuned clcassifier
tuned_svc=SVC(C=10,gamma=.01,kernel='rbf')
score_cross_val_s=cross_val_score(tuned_svc,x_train,y_train,cv=10)
tuned_svc.fit(x_train,y_train)
predict = tuned_svc.predict(x_train)
lm= metrics.confusion_matrix(y_train, predict)


print("==================================Support Vector Machine=============================== ")
print("mean cross validation score with standard deviation")
print(score_cross_val_s.mean(),'+/- ',score_cross_val_s.std())
print("confusion matrix")
print(lm)





#Random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()

#print(rf.get_params())

estimator_options=[50,100,150,200]
param_grid=dict(n_estimators=estimator_options)


grid=GridSearchCV(rf,param_grid,cv=10,scoring='accuracy')

grid.fit(x_train,y_train)
print(grid.best_score_)

print(grid.best_params_)

tuned_rf=RandomForestClassifier(n_estimators=150)
cross_val_score_rf=cross_val_score(tuned_rf,x_train,y_train,cv=10)

tuned_rf.fit(x_train,y_train)

predict_rf=tuned_rf.predict(x_train)

confusion_matrix_rf=metrics.confusion_matrix(y_train,predict_rf)




print("==================================Random Forest =============================== ")
print("mean cross validation score with standard deviation")
print(cross_val_score_rf.mean(),'+/- ',cross_val_score_rf.std())
print("confusion matrix")
print(confusion_matrix_rf)



#Bagged Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

cart=DecisionTreeClassifier()

bc=BaggingClassifier(base_estimator=cart)
estimator_options=[50,100,150,200]
param_grid=dict(n_estimators=estimator_options)

grid=GridSearchCV(bc,param_grid,cv=10,scoring='accuracy')
grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)

tuned_bc=BaggingClassifier(base_estimator=cart,n_estimators=100)
cross_val_score_bc=cross_val_score(tuned_bc,x_train,y_train,cv=10)

tuned_bc.fit(x_train,y_train)
predict_bc=tuned_bc.predict(x_train)
confusion_mat_bc=metrics.confusion_matrix(y_train,predict_bc)


print("==================================Bagging Decision Tree Classifier=============================== ")
print("mean cross validation score with standard deviation")
print(cross_val_score_bc.mean(),'+/- ',cross_val_score_bc.std())
print("confusion matrix")
print(confusion_mat_bc)



#Extra Tree classifier
from sklearn.ensemble import ExtraTreesClassifier

et=ExtraTreesClassifier()

estimator_options=[50,100,150,200]

param_grid=dict(n_estimators=estimator_options)
grid=GridSearchCV(et,param_grid,cv=10,scoring='accuracy')

grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)

tuned_et=ExtraTreesClassifier(n_estimators=150)
cross_val_score_et=cross_val_score(tuned_et,x_train,y_train,cv=10)
tuned_et.fit(x_train,y_train)

predict_et=tuned_et.predict(x_train)

confusion_mat_et=metrics.confusion_matrix(y_train,predict_et)


print("==================================Extra Trees Classifier=============================== ")
print("mean cross validation score with standard deviation")
print(cross_val_score_et.mean(),'+/- ',cross_val_score_et.std())
print("confusion matrix")
print(confusion_mat_et)


#Ada boost classifier
from sklearn.ensemble import AdaBoostClassifier

ab=AdaBoostClassifier()

estimator_options=[50,100,150,200]

param_grid=dict(n_estimators=estimator_options)
grid=GridSearchCV(ab,param_grid,cv=10,scoring='accuracy')

grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)

tuned_ab=AdaBoostClassifier(n_estimators=50)
cross_val_score_ab=cross_val_score(tuned_ab,x_train,y_train,cv=10)
tuned_ab.fit(x_train,y_train)

predict_ab=tuned_ab.predict(x_train)

confusion_mat_ab=metrics.confusion_matrix(y_train,predict_et)


print("==================================ADA boost Classifier=============================== ")
print("mean cross validation score with standard deviation")
print(cross_val_score_ab.mean(),'+/- ',cross_val_score_ab.std())

print("confusion matrix")
print(confusion_mat_ab)


#gradient boost classifier
from sklearn.ensemble import GradientBoostingClassifier

gb=GradientBoostingClassifier()

estimator_options=[50,100,150,200]

param_grid=dict(n_estimators=estimator_options)
grid=GridSearchCV(gb,param_grid,cv=10,scoring='accuracy')

grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)

tuned_gb=GradientBoostingClassifier(n_estimators=50)
cross_val_score_gb=cross_val_score(tuned_gb,x_train,y_train,cv=10)
tuned_gb.fit(x_train,y_train)


predict_gb=tuned_gb.predict(x_train)

confusion_mat_gb=metrics.confusion_matrix(y_train,predict_et)


print("==================================Stpchastic Gradient Boosting Classifier=============================== ")
print("mean cross validation score with standard deviation")
print(cross_val_score_gb.mean(),'+/- ',cross_val_score_gb.std())
print("confusion matrix")
print(confusion_mat_gb)


#A voting Ensemble of Logistic Regression, SVM, k-NN and Random Forest classifier

from sklearn.ensemble import VotingClassifier



clf1=tuned_logisticRegr=LogisticRegression(C=0.1,solver = 'lbfgs')
clf2=tuned_svc=SVC(C=10,gamma=.01,kernel='rbf')
clf3=tuned_knn=KNeighborsClassifier(n_neighbors=7)
clf4=tuned_rf=RandomForestClassifier(n_estimators=150)


vc=VotingClassifier(estimators=[('lr',clf1),('svc',clf2),('knn',clf3),('rf',clf4)],voting='hard')

cross_val_score_vc=cross_val_score(vc,x_train,y_train,cv=10)

vc.fit(x_train,y_train)
predict_vc=vc.predict(x_train)
confusion_mat_vc=metrics.confusion_matrix(y_train,predict_vc)


print("==================================voting Classifier=============================== ")
print("mean cross validation score with standard deviation")
print(cross_val_score_vc.mean(),'+/- ',cross_val_score_vc.std())
print("confusion matrix")
print(confusion_mat_vc)





