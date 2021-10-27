import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle

df = pd.read_csv('../data/breast_cancer.csv')
df.drop(df.columns[[0, -1]], axis=1, inplace=True)

y = df['diagnosis']
X = df.drop(['diagnosis'], axis=1)

# encode the target value
y = np.asarray([1 if row == 'M' else 0 for row in y])

X = X[['concave points_mean', 'area_mean', 'radius_mean', 'perimeter_mean', 'concavity_mean']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))


model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)

print(clf_report)

pickle.dump(model, open('../app/cancer_model.pkl', 'wb'))
