import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc

data = pd.read_csv("creditcard.csv")
print(data.iloc[0])
print(data.Class.value_counts())

##standardizing the data
data["Normalized Amount"] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data.drop(["Time","Amount"],axis=1,inplace=True)
print(data.head())

#clf = RandomForestClassifier(n_estimators=100)

def data_prepration(x): # preparing data for training and testing as we are going to use different data
    #again and again so make a function
    x_features= x.iloc[:,x.columns != "Class"]
    x_labels=x.iloc[:,x.columns=="Class"]
    x_features_train,x_features_test,x_labels_train,x_labels_test = train_test_split(x_features,x_labels,test_size=0.3)
    print("length of training data")
    print(len(x_features_train))
    print("length of test data")
    print(len(x_features_test))
    return(x_features_train,x_features_test,x_labels_train,x_labels_test)

data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data)


print("Test\n",data_labels_test.Class.value_counts())
print("Train\n",data_labels_train.Class.value_counts())

clf=RandomForestClassifier(n_estimators=100, random_state=12)
clf.fit(data_features_train, data_labels_train.values.ravel())

print("HELLO")
print(clf.score(data_features_test, data_labels_test))

feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = data_features_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)

print(feature_importances)

data1=data[["V14","V10","V12","V17","V11","Class"]]
data1.head()

data_features_train,data_features_test,data_labels_train,data_labels_test=data_prepration(data1)

x_train, x_val, y_train, y_val = train_test_split(data_features_train,data_labels_train,
                                                  test_size = .1,
                                                  random_state=12)


sm = SMOTE(random_state=12, ratio = 1.0)

print("SM")
print(sm)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.values.ravel())


clf_rf = RandomForestClassifier(n_estimators=100, random_state=12)
clf_rf.fit(x_train_res, y_train_res)

print('Validation Results RF')
print(clf_rf.score(x_val, y_val))
print(recall_score(y_val, clf_rf.predict(x_val)))
print(precision_score(y_val, clf_rf.predict(x_val)))
print('\nTest Results')
print(clf_rf.score(data_features_test, data_labels_test))
print(recall_score(data_labels_test, clf_rf.predict(data_features_test)))
print(precision_score(data_labels_test, clf_rf.predict(data_features_test)))

rus = RandomUnderSampler(random_state=12)
x_train_res, y_train_res = rus.fit_sample(x_train, np.array(y_train.iloc[:,0]))

clf_rf.fit(x_train_res, y_train_res)

print('Validation Results')
print(clf_rf.score(x_val, y_val))
print(recall_score(y_val, clf_rf.predict(x_val)))
print(precision_score(y_val, clf_rf.predict(x_val)))

print('\nTest Results')
print(clf_rf.score(data_features_test, data_labels_test))
print(recall_score(data_labels_test, clf_rf.predict(data_features_test)))
print(precision_score(data_labels_test, clf_rf.predict(data_features_test)))


print("END")
bbc = BalancedBaggingClassifier(random_state=12)
bbc.fit(x_train, np.array(y_train.iloc[:,0]))

print('Validation Results')
print(bbc.score(x_val, y_val))
print(recall_score(y_val, bbc.predict(x_val)))
print(precision_score(y_val, bbc.predict(x_val)))
print('\nTest Results')
print(bbc.score(data_features_test, data_labels_test))
print(recall_score(data_labels_test, bbc.predict(data_features_test)))
print(precision_score(data_labels_test, bbc.predict(data_features_test)))




clf_xg = GradientBoostingClassifier(learning_rate=0.15, n_estimators=70, min_samples_split=0.5, min_samples_leaf=45, max_depth=8,max_features ='sqrt',subsample =0.8)
clf_xg.fit(x_train_res, y_train_res)

print('Validation Results RF')
print(clf_xg.score(x_val, y_val))
print(recall_score(y_val, clf_xg.predict(x_val)))
print(precision_score(y_val, clf_xg.predict(x_val)))
print('\nTest Results')
print(clf_xg.score(data_features_test, data_labels_test))
print(recall_score(data_labels_test, clf_xg.predict(data_features_test)))
print(precision_score(data_labels_test, clf_xg.predict(data_features_test)))

# print("true ending")

C_param_range = [0.001,0.01,0.1,1,10,100]


for i in C_param_range:
    clf_lr = LogisticRegression(penalty = 'l2', C = i, random_state=0)
    clf_lr.fit(x_train_res, y_train_res)
    print('Validation Results C Values:', i)
    print(clf_lr.score(x_val, y_val))
    print(recall_score(y_val, clf_lr.predict(x_val)))
    print(precision_score(y_val, clf_lr.predict(x_val)))
    print('\nTest Results')
    print(clf_lr.score(data_features_test, data_labels_test))
    print(recall_score(data_labels_test, clf_lr.predict(data_features_test)))
    print(precision_score(data_labels_test, clf_lr.predict(data_features_test)))




for i in C_param_range:
    clf_lrb = LogisticRegression(class_weight='balanced', C = i,penalty = 'l2')
    clf_lrb.fit(x_train, np.array(y_train.iloc[:, 0]))
    print('Validation Results RF')
    print(clf_lrb.score(x_val, y_val))
    print(recall_score(y_val, clf_lrb.predict(x_val)))
    print(precision_score(y_val, clf_lrb.predict(x_val)))
    print('\nTest Results')
    print(clf_lrb.score(data_features_test, data_labels_test))
    print(recall_score(data_labels_test, clf_lrb.predict(data_features_test)))
    print(precision_score(data_labels_test, clf_lrb.predict(data_features_test)))