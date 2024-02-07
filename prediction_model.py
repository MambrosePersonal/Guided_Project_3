import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('troop_movements.csv')
#print(data)

df_all = pd.DataFrame(df, columns = ["timestamp", "unit_id", "unit_type","empire_or_resistance", "location_x",
                                   "location_y", "destination_x", "destination_y","homeworld"])
# print(df_all)


df_evr = df.groupby('empire_or_resistance').size().reset_index(name="count")

# print(df_evr) 

df_hw = df.groupby('homeworld').size().reset_index(name="count")

# print(df_hw)

df_ut = df.groupby('unit_type').size().reset_index(name="count")

# print(df_ut)

df['is_resistance'] = df['empire_or_resistance'] == 'resistance'

# print(df.head())
ax = sns.barplot(x = df['empire_or_resistance'].value_counts(), y = df['empire_or_resistance'].value_counts() , data = df)

ax.set_xticklabels(['Resistance', 'Empire'])
plt.xlabel('Empire or Resistance')
plt.ylabel('Count')
plt.title('Character Count by Empire vs Resistance')
plt.show()



X = df[['homeworld', 'unit_type']]
y = df['empire_or_resistance']

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#creates bar plot
importances = model.feature_importances_

print(importances)

feature_importances = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})

feature_importances = feature_importances.sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()

with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)