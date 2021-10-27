import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle


df = pd.read_csv('cleaned_liver_df.csv')


y = df['Dataset']
X = df.drop('Dataset', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    shuffle=True, random_state=42)

print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)

print(clf_report)

pickle.dump(model, open('../app/liver_model.pkl', 'wb'))
