import pandas as pd   
import numpy as np
import time
import shap
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


start = time.time()

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV

## random forest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50,criterion="gini", max_depth=16, min_samples_leaf=5, max_leaf_nodes=10)

NF_data = pd.read_csv("data/NF-ToN-IoT.csv")
nf_X = NF_data.drop(axis=1, columns=['Attack'])
nf_X = nf_X.drop(axis=1, columns=['Label'])
nf_X = nf_X.drop(axis=1, columns=['IPV4_SRC_ADDR'])
nf_X = nf_X.drop(axis=1, columns=['IPV4_DST_ADDR'])
# Adjust sample range as necessary, this selection is used as it provides a good range of every classification without taking a long time to process
nf_X = nf_X.loc[766450:768950]

nf_Y = NF_data['Attack']
nf_Y = pd.DataFrame(nf_Y)
nf_Y = nf_Y.loc[766450:768950]




X_train, X_test, y_train, y_test = train_test_split(nf_X, nf_Y, random_state=1, shuffle=True)


# GridSearch params
params = {'n_estimators': [50,100,150,200],
 'criterion': ['gini', 'entropy'],
 'max_depth': range(1, 16),
 'min_samples_leaf': range(0, 25, 5)[1:]}

# Uncomment the below line to use GridSearch to find optimal classifier params for your dataset
#classifier = GridSearchCV(param_grid=params, estimator=RandomForestClassifier(random_state=10),return_train_score=True, cv=10)

# Fitting classifier to the Training set    
classifier.fit(X_train, y_train)

#print(classifier.best_params_)
#print(classifier.best_params_)
# Predicting the Test set results
y_pred = classifier.predict(X_test)


df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
df_y_pred.to_csv("data/y_pred.csv",encoding='utf-8')   

# Performance metrics
from sklearn.metrics import confusion_matrix, accuracy_score, multilabel_confusion_matrix 
from sklearn.metrics import precision_recall_fscore_support
cm = confusion_matrix(y_test, y_pred)
df_y1_test = pd.DataFrame(y_test)
df_y1_test.to_csv("data/y1_test.csv", encoding = 'utf-8')

#accuracy -number of instance correctly classified
acsc = accuracy_score(y_test, y_pred) 
df_cm = pd.DataFrame([[cm[1][1], cm[0][0],cm[0][1], cm[1][0]]], 
                        index=[0],
                        columns=['True Positives','True Negatives', 'False Positives', 'False Negatives'])
print(df_cm)
#precision, recall, fscore, support
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred,average='weighted', labels=np.unique(y_pred))






prc_auc = ''
df_metrics = pd.DataFrame([[acsc, precision, recall, fscore]], 
                        index=[0],
                        columns=['accuracy','precision', 'recall', 'fscore'])



shap.initjs()
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test)
#expected_value = explainer.expected_value[2]
#print(classifier.classes_)
#select = range(20)
#features = X_test.iloc[select]
#shap_values = explainer.shap_values(features)[1]
shap.summary_plot(shap_values, X_test, max_display = 39, class_names=classifier.classes_)
#expected_value = explainer.expected_value
#class_names= ['Analysis','Backdoor', 'DoS', 'Exploits', 'Fuzzers', 'Generic', 'Normal', 'Reconnaisance', 'Shellcode', Worms]


#shap.decision_plot(expected_value, explainer.shap_values(features)[2], feature_names=['FC1_Read_Input_Register','FC2_Read_Discrete_Value','FC3_Read_Holding_Register', 'FC4_Read_Coil' ], feature_display_range = range(10))
#shap.summary_plot(shap_values[3], X_test)

print(df_metrics)
end = time.time()

print(df_metrics.iloc[0][0],',',df_metrics.iloc[0][1],',',df_metrics.iloc[0][2],',',df_metrics.iloc[0][3],',',df_cm.iloc[0][0],',',df_cm.iloc[0][1],',',df_cm.iloc[0][2],',',df_cm.iloc[0][3],',', end-start)

#importances = classifier.feature_importances_
#feature_names = X.columns
#indices = np.argsort(importances)
#indices = np.flip(indices, axis=0)
#indices = indices[:40]

#for i in indices:
# print(feature_names[i], ':', importances[i])

#print(classification_report(y1_test, y_pred))

print("Time taken:", end-start)

