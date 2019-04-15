import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def test_model(data,column,model,TFIDF_column):
    y = data[column]
    X = data[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = model()
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    y_pred = model.predict_proba(X_test)

    print("Accuracy score of this model is" + "\n")
    print(accuracy_score(y_test,y_predict))
    print("\n" + "Confusion Matrix of this model is" + "\n")
    print(confusion_matrix(y_test,y_predict))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(log_loss(y_test,y_pred))
    print("\n" + "AUC score of this model is" + "\n")
    print(roc_auc_score(y_test,y_predict))
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")
    return None

def test_model2(data,column,model,TFIDF_column):
    y = data[column]
    X = data[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = model()
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)

    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")
    return None

def test_model3(data,column,model,TFIDF_column):
    data2 = data.sample(frac=1)
    data2_training = data2[:16000]
    data2_testing = data2[16000:]
    y = data2_training[column]
    X = data2_training[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = model()
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for initial test (first 80% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    print("\n")

    y = data2_testing[column]
    X = data2_testing[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #model = model()
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for final test (last 20% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")
    return None

def test_model4(data,column,TFIDF_column):
    data2 = data.sample(frac=1)
    data2_training = data2[:16000]
    data2_testing = data2[16000:]
    y = data2_training[column]
    X = data2_training[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = RandomForestClassifier(n_estimators=500)
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for initial test (first 80% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    print("\n")

    y = data2_testing[column]
    X = data2_testing[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #model = model()
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for final test (last 20% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")
    return model

def test_model5(data,column,TFIDF_column):
    data2 = data.sample(frac=1)
    data2_training = data2[:16000]
    data2_testing = data2[16000:]
    y = data2_training[column]
    X = data2_training[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = GradientBoostingClassifier(subsample=0.5, learning_rate=0.01)
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for initial test (first 80% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    print("\n")

    y = data2_testing[column]
    X = data2_testing[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    #model = model()
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for final test (last 20% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")
    return None

def test_model4b(data,column,TFIDF_column):
    data2 = data.sample(frac=1)
    data2_training = data2[:16000]
    data2_testing = data2[16000:]
    y = data2_training[column]
    X = data2_training[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]
    X_train_, X_test_, y_train_, y_test_ = train_test_split(X, y, test_size=0.25)
    model = RandomForestClassifier(n_estimators=500)
    kf = KFold(n_splits=5, shuffle=True)

    ll_performance = []
    auc_performance = []
    acc_performance = []
#kfold split on X_ (which is X_train of len 110)
    for train_index, test_index in kf.split(X_train_):
        X_train, X_test = X_train_.iloc[train_index], X_train_.iloc[test_index]
        y_train, y_test = y_train_.iloc[train_index], y_train_.iloc[test_index]
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
        log_ll = log_loss(y_test, y_pred)
        ll_performance.append(log_ll)
        auc = roc_auc_score(y_test,y_predict)
        auc_performance.append(auc)
        acc = accuracy_score(y_test,y_predict)
        acc_performance.append(acc)
        
        
    print("\n" + "Score summary for initial test (first 80% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(np.mean(acc_performance))
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(np.mean(ll_performance))
    print("\n" + "AUC score of this model is" + "\n")
    print(np.mean(auc_performance))
    print("\n")

    y_final = data2_testing[column]
    X_final = data2_testing[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none'] + TFIDF_column]

#kfold split on X_ (which is X_train of len 110)
    model.fit(X_train_, y_train_)
    y_predict = model.predict(X_test_)
    y_pred = model.predict_proba(X_test_)
    log_ll = log_loss(y_test_, y_pred)
    auc = roc_auc_score(y_test_,y_predict)
    acc = accuracy_score(y_test_,y_predict)
        
    print("\n" + "Score summary for final test (last 20% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(acc)
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(log_ll)
    print("\n" + "AUC score of this model is" + "\n")
    print(auc)
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")

#kfold split on X_ (which is X_train of len 110)
    model.fit(X, y)
    y_predict = model.predict(X_final)
    y_pred = model.predict_proba(X_final)
    log_ll = log_loss(y_final, y_pred)
    auc = roc_auc_score(y_final,y_predict)
    acc = accuracy_score(y_final,y_predict)
        
    print("\n" + "Score summary for final test (last 20% of data)" + "\n")
    print("\n" + "Accuracy score of this model is" + "\n")
    print(acc)
    print("\n" + "Log-Loss score of this model is" + "\n")
    print(log_ll)
    print("\n" + "AUC score of this model is" + "\n")
    print(auc)
    #print("\n" + "Coefficients of this model are" + "\n")
    #print(model.coef_)
    #print("\n" + "Most effective predictor was" + "\n")
    
    return model