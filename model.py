import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pickle

results=pd.read_csv("BD_cadeaux (2).csv")
results=results[results.Cadeau!= "autre"]
dummies=pd.get_dummies(results.Age)
merged=pd.concat([results,dummies],axis='columns')
results=merged.drop(['Age','<18'],axis='columns')
dummies2=pd.get_dummies(results.RorP)
merged2=pd.concat([results,dummies2],axis='columns')
results=merged2.drop(['RorP','Romantique'],axis='columns')
dummies3=pd.get_dummies(results.CoS)
merged3=pd.concat([results,dummies3],axis='columns')
results=merged3.drop(['CoS','Cool'],axis='columns')
dummies4=pd.get_dummies(results.EorI)
merged4=pd.concat([results,dummies4],axis='columns')
results=merged4.drop(['EorI','Introverti'],axis='columns')
dummies5=pd.get_dummies(results.Occasion)
merged5=pd.concat([results,dummies5],axis='columns')
results=merged5.drop(['Occasion','Évènement spécial'],axis='columns')
dummies6=pd.get_dummies(results.Catégorie)
merged6=pd.concat([results,dummies6],axis='columns')
results=merged6.drop(['Catégorie','Utile'],axis='columns')
dummies7=pd.get_dummies(results.Budget_DH)
merged7=pd.concat([results,dummies7],axis='columns')
results=merged7.drop(['Budget_DH','>10000'],axis='columns')
dummies8=pd.get_dummies(results.Relation)
merged8=pd.concat([results,dummies8],axis='columns')
results=merged8.drop(['Relation','familiale'],axis='columns')
#X_train,Y_train,X_test,Y_test
X=results.drop(columns='Cadeau',axis=1) 
Y=results['Cadeau']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
model_tree=DecisionTreeClassifier(class_weight='balanced')
model_tree.fit(X_train,Y_train)
pickle.dump(model_tree,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(X)


