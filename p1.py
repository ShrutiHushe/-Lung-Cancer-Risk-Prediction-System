#import lib

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

#load the data
data = pd.read_csv("lc_march24.csv")
print(data)

#check for null data
print(data.isnull().sum())

#check for dupliacte data
print(data.duplicated().sum())
data.drop_duplicates(keep="first", inplace=True)
print(data.duplicated().sum())

#features and target
features = data.drop("LUNG_CANCER", axis="columns")
target = data["LUNG_CANCER"]

#handle cat data
cfeatures = pd.get_dummies(features)
print(features)
print(cfeatures)

#feature scaling
mms = MinMaxScaler()
nfeatures = mms.fit_transform(cfeatures)
print(nfeatures)

#train and test
x_train, x_test, y_train, y_test = train_test_split(nfeatures, target)

#model
model = KNeighborsClassifier(n_neighbors=75, metric="euclidean")
model.fit(x_train, y_train)

#confusion matrix
cm = confusion_matrix(y_test, model.predict(x_test))
print(cm)

#confusiom report
cr = classification_report(y_test, model.predict(x_test))
print(cr)

#prediction
age = float(input("enter age"))
smoking = int(input("Smoking: 1 for N) and 2 for YES"))
yellow_fingers = int(input("yellow_fingers: 1 for N) and 2 for YES"))
anxiety = int(input("anxiety: 1 for N) and 2 for YES"))
peer_pressure = int(input("peer_pressure: 1 for N) and 2 for YES"))
chronic_disease = int(input("chronic_disease: 1 for N) and 2 for YES"))
fatique = int(input("fatique: 1 for N) and 2 for YES"))
allergy = int(input("allergy: 1 for N) and 2 for YES"))
wheezing = int(input("wheezing: 1 for N) and 2 for YES"))
alcohol = int(input("alcohol: 1 for N) and 2 for YES"))
coughing = int(input("coughing: 1 for N) and 2 for YES"))
sob = int(input("sob: 1 for N) and 2 for YES"))
sd = int(input("sd: 1 for N) and 2 for YES"))
cp= int(input("cp: 1 for N) and 2 for YES"))
gender = int(input("1 for female and 2 for male"))

if gender == 1:
    d = [[age,smoking,yellow_fingers,anxiety,peer_pressure,chronic_disease,fatique,allergy,wheezing,alcohol,coughing,sob,sd,cp,1,0]]
    
else:
   d = [[age,smoking,yellow_fingers,anxiety,peer_pressure,chronic_disease,fatique,allergy,wheezing,alcohol,coughing,sob,sd,cp,0,1]]  

nd = mms.transform(d)
ans = model.predict(nd)
print(ans)