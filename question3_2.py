from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Logistic Regression
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
print(f"Logistic Regression Accuracy: {log_reg_acc}")
print(classification_report(y_test, y_pred_log_reg))

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf_clf = rf_clf.predict(X_test)
rf_clf_acc = accuracy_score(y_test, y_pred_rf_clf)
print(f"Random Forest Classifier Accuracy: {rf_clf_acc}")
print(classification_report(y_test, y_pred_rf_clf))

# Support Vector Machine
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
y_pred_svm_clf = svm_clf.predict(X_test)
svm_clf_acc = accuracy_score(y_test, y_pred_svm_clf)
print(f"SVM Classifier Accuracy: {svm_clf_acc}")
print(classification_report(y_test, y_pred_svm_clf))

# Determine the best model
best_model = None
best_acc = max(log_reg_acc, rf_clf_acc, svm_clf_acc)
if best_acc == log_reg_acc:
    best_model = log_reg
elif best_acc == rf_clf_acc:
    best_model = rf_clf
else:
    best_model = svm_clf

print(f"Best Model: {best_model}")

joblib.dump(best_model, 'best_heart_disease_model.pkl')

print("Model training and evaluation completed. Best model saved to disk.")