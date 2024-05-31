import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df1=pd.read_csv('/content/HR-Employee-Attrition.csv')

df1.head()

X = df1.drop('Attrition', axis=1)

y = df1['Attrition']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

non_numerical_cols = X_train.select_dtypes(include=["object"]).columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in non_numerical_cols:
    X_train[col] = le.fit_transform(X_train[col])

for col in non_numerical_cols:
    X_test[col] = le.fit_transform(X_test[col])

model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
