from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

import pandas as pd

dataset = pd.read_csv('your file csv')
X = dataset.drop(columns=['drop kolom yang tidak perlu']).values
y = dataset.Kelas.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
svc = SVC(kernel='rbf', max_iter=1000, gamma=0.1,
          C=2).fit(X_train, y_train)
y_pred = svc.predict(X_test)

# importing confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)

# importing accuracy_score, precision_score, recall_score, f1_score
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(
    precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(
    recall_score(y_test, y_pred, average='micro')))
print(
    'Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(
    precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(
    recall_score(y_test, y_pred, average='macro')))
print(
    'Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(
    precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(
    recall_score(y_test, y_pred, average='weighted')))
print(
    'Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

print('\nClassification Report\n')
print(classification_report(y_test, y_pred,
                            target_names=['Rendah', 'Sangat Tinggi', 'Sedang', 'Tinggi']))
