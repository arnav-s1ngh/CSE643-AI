from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import pandas as pd
from ucimlrepo import fetch_ucirepo
#from the ucimlrepo description
abalone=fetch_ucirepo(id=1)
X=abalone.data.features
y=abalone.data.targets
#have to process categorical variables
X=pd.get_dummies(X, columns=["Sex"])
# divide dataset
#17 gives best division out of all the ints from 0-100 i have tried yet
X_train,X_vt,y_train,y_vt =train_test_split(X,y,test_size=0.3,random_state=17)
X_val,X_test,y_val,y_test=train_test_split(X_vt,y_vt,test_size=0.5,random_state=17)
#make a chain model
lr_model=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())
lr_model.fit(X_train,y_train)
#make the predictions
y_pred=lr_model.predict(X_train)
fevalr2=r2_score(y_train,y_pred)
#cross validate
cvres=cross_val_score(lr_model,X_val,y_val,cv=15,scoring='r2')
cvmean=sum(cvres)/15
cvst=np.std(cvres)
#print
print(f"Full dataset train and eval R2 score: {fevalr2}")
print(f"70-15-15 Cross validation boxplot: mean={cvmean}, std={cvst}")