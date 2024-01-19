import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#dataset importation
dataset=pd.read_csv("heart.csv")

#Data analysis
"""
DESCRIPTION DU DATASET : (303, 14)


1)age= age of the patient in year
2)sex= gender of the patient (1 = homme ; 0 = femme)
3)cp=  type de douleur thoracique (0 = angine typique ; 1 = angine atypique ;
    2 = douleur non angineuse ; 3 = asymptomatique)
4)trtbps=  tension artérielle au repos (en mm de Hg à l'admission à l'hôpital)
5)chol= cholestérol sérique en mg/dl récupéré via le capteur IMC ou autre
6)fbs=glycémie à jeun > 120 mg/dl (1 = vrai ; 0 = faux)
7)restecg=résultats de l'électrocardiographie au repos (0 = normal ;
    1 = présence de ST-T ; 2 = hypertrophie)
8)thalachh=fréquence cardiaque maximale atteinte
9)exng= angine/douleur déclenchée par l'exercice (1 = oui ; 0 = non)
10)\oldpeak - dépression du segment ST induite par l'exercice par rapport au repos
11)\slope - la pente du segment ST lors de l'exercice de pointe (1 = ascendante ;
     2 = plate ; 3 = descendante)
12)caa= nombre de principaux vaisseaux (de 0 à 4) colorés par fluoroscopie
13)\thal - 1 = normal ; 2 = défaut fixe ; 3 = défaut réversible
14)output= l'attribut prédit - diagnostic de maladie cardiaque (statut de la maladie angiographique) 
(Valeur 0 = rétrécissement du diamètre < 50% ; Valeur 1 = rétrécissement du diamètre > 50%)
0=faible probabilite d'avoir une attaque, 1=forte probabilité d'avoir une attaque
"""

#verifions s'il ya des valeurs manquantes
missing_values = dataset.isna().sum()


#correlation textuelle et graphique

cor=dataset.corr()
sns.heatmap(cor)
plt.savefig('heatmap.png', dpi=250)

df=dataset.drop(["oldpeak","slp","thall"],axis=1)
a=df.describe()

cor2=df.corr()
sns.heatmap(cor2) #cool aucune variable corrélé ! pas de multicolinéarité super


##nous faisons une analyse descriptive uni et bivariée les uns des autres
##data analysis
plt.figure(figsize=(20,10))
plt.title("age des patients")
plt.xlabel("age")
sns.countplot(x='age',data=df)#effectif plus grand age compris entre 51 et 67


plt.figure(dpi=500)
plt.title("sexe of patients  0=femele, 1=male")
sns.countplot(x='sex',data=df)#plus d'homme que de femme dans la population


cp_data=df['cp'].value_counts().reset_index()
cp_data ["index"][3]="asymptomatique"
cp_data ["index"][2]="douleur non anginieuse"
cp_data ["index"][1]="angine atypique"
cp_data ["index"][0]="angine typique"
cp_data
plt.figure(figsize=(20,10))
plt.title("type de douleur thoracique")
sns.barplot(x="index", y="cp" ,data=cp_data) #affiche la var cp par nombre de modalité 


ecg_data=df['fbs'].value_counts().reset_index()
ecg_data ["index"][1]="OUI glycémie à jeun supérieure à 120 mg/dl"
ecg_data ["index"][0]="NON glycémie à jeun pas supérieure à 120 mg/dl"
ecg_data
plt.figure(figsize=(20,10))
plt.title("glycémie à jeun")
sns.barplot(x="index", y="fbs" ,data=ecg_data) #affiche la var fbs par nombre de modalité 


ecg_data=df['restecg'].value_counts().reset_index()
ecg_data ["index"][2]="hypertrophie"
ecg_data ["index"][1]="présence ST-T"
ecg_data ["index"][0]="normal"
ecg_data
plt.figure(figsize=(20,10))
plt.title("résultats de l'électrocardiographie au repos")
sns.barplot(x="index", y="restecg" ,data=ecg_data) #affiche la var restecg par nombre de modalité 




restecg_data=df['restecg'].value_counts().reset_index()
restecg_data ["index"][2]="hypertrophie"
restecg_data ["index"][1]="présence ST-T"
restecg_data ["index"][0]="normal"
restecg_data
plt.figure(figsize=(20,10))
plt.title("résultats de l'électrocardiographie au repos")
sns.barplot(x="index", y="restecg" ,data=restecg_data) #affiche la var restecg par nombre de modalité 




caa_data=df['caa'].value_counts().reset_index()
caa_data ["index"][4]="04 vaisseaux colorés "
caa_data ["index"][3]="03 vaisseaux colorés "
caa_data ["index"][2]="02 vaisseaux coloré "
caa_data ["index"][1]="01 vaisseau coloré "
caa_data ["index"][0]="aucun vaisseau coloré"
caa_data
plt.figure(figsize=(20,10))
plt.title("nombre de principaux vaisseaux colorés par fluoroscopie")
sns.barplot(x="index", y="caa" ,data=caa_data) #affiche la var caa par nombre de modalité 


#pour les variables continues (verifions en passant la normalité des observations)
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.displot(df['trtbps'],kde=True,color="black")
plt.xlabel("tension artérielle au repos (en mm Hg à l'admission à l'hôpital)")

plt.subplot(1,2,2)
sns.displot(df['thalachh'],kde=True,color="teal")
plt.xlabel("fréquence cardiaque maximale atteinte")

plt.figure(figsize=(10,10))
sns.displot(df['chol'],kde=True,color="red")
plt.xlabel("cholestérol sérique en mg/dl récupéré via le capteur IMC ou autre")


sns.pairplot(df,hue="output") #éffectue une analyse bivarié en matrice de scatter plot



dataset.columns

df=pd.DataFrame(df,columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 
                'thalachh','exng', 'caa', 'output'])

# division du dataset

X=df.iloc[:, :-1] #var independante iloc (fonction de pd) vas recuperer les indices dont on aura besoin (: pour toute et :-1 pour toute sauf la dernière) 
y=df.iloc[:, -1] # -1 pour la dernière

#feature scaling
"""pour eviter que les grandes valeurs n'écrase pas les plus petites durant 
 l'entrainement du model
  deux methode de scaling existe: la STANDARDISATION(cen) et la NORMALISATION 
 la STANDARDISATION: loi normal centré reduite (-moyenne divisé par l'ecart-type)
 la NORMALISATION: -min(x)divisé par max(x)-min(x)
"""
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)


###creation du model
"""
il est possible pour notre travail de le passer passer aux models suivant:
   1* regression logistique
   2* decision tree /arbre de decision
   3* svm
   4* knn
   5* random forest
   6* reg multiple
"""

 ## 1)regression logistique

# Construction du modèle
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression()
model_log.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_log.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
a=accuracy*100
a
 #notre model est fiable a environ 85.7142%
 
#pour la prediction d'une valeur souhaité on fera
model_log.predict(np.array([[63,1,3,145,233,1,0,150,0,0]]))
aaa=model_log.predict_proba(np.array([[100,1,3,145,233,1,0,150,0,0]]))

model_log.predict(np.array([[-0.26098,0.681005,	1.00258,	2.306,	-0.9134,	2.39444,	0.898962,	0.540209,	-0.696631,	-0.714429]]))
aaa=model_log.predict_proba(np.array([[100,1,3,145,233,1,0,150,0,0]]))


probabilities = model_log.predict_proba(X_test)
probabilities2 = model_log.predict(X_test)

proba_classe_0 = probabilities[0][0]  # Probabilité de la classe 0
proba_classe_1 = probabilities[0][1]  # Probabilité de la classe 1


# Save the trained model as a pickle string.
import pickle 
saved_model = pickle.dump(model_log, open('heart84.6.pickle','wb'))


 ## 2)arbre de decision/tree decision
# Construction du modèle
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_tree.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
b=accuracy*100 #notre model est fiable a environ 69%
prediction_df=pd.DataFrame({"y_test":y_test,"y_pred":y_pred})


 ## 3)random forest
# Construction du modèle
from sklearn.ensemble import RandomForestClassifier
model_forest = RandomForestClassifier()
model_forest.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_forest.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm= confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
c=accuracy*100
 #notre model est fiable a environ 80%

 ## 4)k plus proche voisin

# Construction du modèle
from sklearn.neighbors import KNeighborsClassifier

error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred=knn.predict(X_test)
    error_rate.append(np.mean(pred !=y_test))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color="blue",linestyle="dashed",marker='o',
         markerfacecolor="red",markersize=10)
plt.xlabel("k value")
plt.ylabel("error rate")
plt.title("obtention de la mailleure valeur de k")
plt.show()    
#d'apres le graphe nous pouvons considerer k=12 comme sa valeur obtimale (mais devrais plutot etre 27)

model_knn = KNeighborsClassifier(n_neighbors=27)
model_knn.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_knn.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
d=accuracy*100
 #notre model est fiable a environ 85.71%


 
 ## 5)SVM linear
# Construction du modèle
from sklearn import svm
model_svm = svm.SVC()
model_svm.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_svm.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm= confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
e=accuracy*100 #notre model est fiable a environ 80,2%



 ## 6)reg_lin_mul
# Construction du modèle
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_svm.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm= confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
f=accuracy*100


#comparaison des accuracy
model_acc=pd.DataFrame({'Model':['reg_log','arbre de decision','random forest','knn',
                      'linear svm','reg_lin_mul'],"accuracy":[a,b,c,d,e,f]})
model_acc=model_acc.sort_values(by=['accuracy'],ascending=False)






###verifions les hyperparamètres/*overfitching* pour notre reg_log
 ##CV
from sklearn.model_selection import GridSearchCV
model_acc 

#la reg_log etant le meilleur model commençons par nous attarder sur lui

param_grid={
    'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
    'penalty':['none','L1','L2','elasticnet'],
    'C':[100,10,1.0,0.01]
    }

grid1=GridSearchCV(LogisticRegression(),param_grid)

grid1.fit(X_train,y_train)

grid1.best_params_
#cool appliquons lès a notre modele de reg_log
#{'C': 100, 'penalty': 'none', 'solver': 'newton-cg'}

 ## 1-1)regression logistique

# Construction du modèle
from sklearn.linear_model import LogisticRegression
model_log = LogisticRegression(C=100, penalty='none', solver= 'newton-cg')
model_log.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_log.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy*100
 #oups rien n'a changé !!!!!
 

 ## 4-4)k plus proche voisin
n_neighbors=range(1,21,2)
weights=['uniform','distance']
metric=['euclidean','manhattan','minkowski']
grid=dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
from sklearn.model_selection import RepeatedStratifiedKFold
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
grid_search=GridSearchCV(estimator=knn,param_grid=grid,n_jobs=-1,cv=cv,
                         scoring='accuracy',error_score=0)
grid_search.fit(X_train,y_train)
grid_search.best_params_ #cool hyperparamètre trouvé
# {'metric': 'manhattan', 'n_neighbors': 11, 'weights': 'distance'}
#appliquons lès

model_knn = KNeighborsClassifier(metric='manhattan',n_neighbors=11,weights='distance')
model_knn.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_knn.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy*100
 #notre model est fiable a environ 85.71% rien ne change tout les paramètre non utilisé!



 ## 5-5)SVM linear
 
kernel=['poly','rbf',"sigmoid"]
C=[50,10,1.0,0.1,0.01]
gamma=['scale']

grid=dict(kernel=kernel,C=C,gamma=gamma)
cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)
grid_search=GridSearchCV(estimator=model_svm,param_grid=grid,n_jobs=-1,cv=cv,scoring='accuracy',error_score=0)

grid_search.fit(X_train,y_train)

grid_search.best_params_ #cool hyperparamètre trouvé
 #{'C': 0.1, 'gamma': 'scale', 'kernel': 'sigmoid'}
 
 
# Construction du modèle
from sklearn.svm import SVC
model_svm = svm.SVC(C= 0.1, gamma= 'scale', kernel= 'sigmoid')
model_svm.fit(X_train, y_train)
# Faire de nouvelles prédictions
y_pred = model_svm.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm= confusion_matrix(y_test, y_pred)
#impression du rapport
rapport= classification_report(y_test,y_pred,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy*100 #notre model quitte de 83,5% à 81%


######verdict final
#apres plusieurs comparaison, reg_log sans réglage d'hyperparamètre is the best 85.7%

###construisons une matrice de confusion appropriée pour notre modèle
 # cm agrandis
options=['disease','No disease']

fig,ax=plt.subplots()
im=ax.imshow(cm,cmap='Set3',interpolation='nearest')

 #montrons tout les cochés
ax.set_xticks(np.arange(len(options)))
ax.set_yticks(np.arange(len(options)))
 #...label
ax.set_xticklabels(options)
ax.set_yticklabels(options)
 #Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

 #loop over data dimensions and create text annotation
for i in range(len(options)):
    for j in range(len(options)):
        text=ax.text(j,i,cm[i,j],ha='center',va='center',color='black')
        
ax.set_title('matrice de confusion du mod reg_log')         
fig.tight_layout()
plt.xlabel('Model prediction')
plt.ylabel("resultat acturl")
plt.show()


















#perspective  
#perspective
#perspective
#perspective
#perspective
#perspective

"""passons maintenant au adaboosting pour obtenir un meilleur accuracy

AdaBoost (Adaptive Boosting) est un algorithme d'apprentissage automatique 
supervisé utilisé pour améliorer les performances des modèles de classification. 
Il a été proposé par Yoav Freund et Robert Schapire en 1996. L'objectif principal
 d'AdaBoost est de combiner plusieurs modèles d'apprentissage faibles 
 (également appelés classifieurs faibles) pour créer un modèle fort capable de 
 bien généraliser et de traiter des problèmes de classification complexes.
"""
from sklearn.ensemble import AdaBoostClassifier

adab=AdaBoostClassifier(base_estimator=model_svm,n_estimators=100,algorithm="SAMME",
                        learning_rate=0.01,random_state=0)
adab.fit(X_train, y_train)
y_pred_adab=adab.predict(X_test)
# Matrice de confusion
from sklearn.metrics import confusion_matrix, classification_report
cm= confusion_matrix(y_test, y_pred_adab)
#impression du rapport
rapport= classification_report(y_test,y_pred_adab,zero_division=True)
#accuracy
from sklearn.metrics import accuracy_score
adab_accuracy = accuracy_score(y_test, y_pred_adab)
adab_accuracy #notre modele a mal fonctionné puisque seulement 51% de precision !






































