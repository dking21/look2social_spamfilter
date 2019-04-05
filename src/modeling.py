from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def test_logistic_regressio(data,column):
    y = data[column]
    X = data[['Docusign', 'onespan', 'signnow','adobe sign',
                           'listed_count', 'statuses_count','followers_count',
                           'favourites_count', 'friends_count','time_float_sin',
                           'time_float_cos', 'is_description_none']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train,y_train)
    y_predict = logistic_model.predict(X_test)
    y_pred = logistic_model.predict_proba(X_test)

    print("Accuracy score of this model is" + "\n")
    print(accuracy_score(y_test,y_predict))
    print("Confusion Matrix of this model is" + "\n")
    print(confusion_matrix(y_test,y_predict))
    print("Log-Loss score of this model is" + "\n")
    print(log_loss(y_test,y_pred))
    print("AUC score of this model is" + "\n")
    print(roc_auc_score(y_test,y_predict))
    return None

def test_model(data,column,model):
    y = data[column]
    X = data[['Docusign', 'onespan', 'signnow','adobe sign','listed_count', 'statuses_count','followers_count','favourites_count', 'friends_count','time_float_sin','time_float_cos', 'is_description_none','text-TF-IDF-0', 'text-TF-IDF-1', 'text-TF-IDF-2',
       'text-TF-IDF-3', 'text-TF-IDF-4', 'text-TF-IDF-5', 'text-TF-IDF-6',
       'text-TF-IDF-7', 'text-TF-IDF-8', 'text-TF-IDF-9', 'text-TF-IDF-10',
       'text-TF-IDF-11', 'text-TF-IDF-12', 'text-TF-IDF-13', 'text-TF-IDF-14',
       'text-TF-IDF-15', 'text-TF-IDF-16', 'text-TF-IDF-17', 'text-TF-IDF-18',
       'text-TF-IDF-19']]
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
