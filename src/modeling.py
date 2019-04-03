from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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