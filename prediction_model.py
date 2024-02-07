import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('troop_movements.csv')

#Read all info into a pandas dataframe and display it
df_all = pd.DataFrame(df, columns = ["timestamp", "unit_id", "unit_type","empire_or_resistance", "location_x",
                                   "location_y", "destination_x", "destination_y","homeworld"])
print(df_all)

# Creates grouped data showing counts of empire vs resistance
df_evr = df.groupby('empire_or_resistance').size().reset_index(name="count")
print(df_evr) 

# Creates grouped data showing counts of characters by homeworld
df_hw = df.groupby('homeworld').size().reset_index(name="count")
print(df_hw)

# Creates grouped data showing counts of characters by unit_type
df_ut = df.groupby('unit_type').size().reset_index(name="count")
print(df_ut)

# Creates a new feature called is_resistance with a True or False value based on empire_or_resiatance
df['is_resistance'] = df['empire_or_resistance'] == 'resistance'
print(df.head())

# Creates a bar plot using Seaborn showing Empire vs Resistance distribution
ax = sns.barplot(x = df['empire_or_resistance'].value_counts(), y = df['empire_or_resistance'].value_counts() , data = df)
ax.set_xticklabels(['Resistance', 'Empire'])
plt.xlabel('Empire or Resistance')
plt.ylabel('Count')
plt.title('Character Count by Empire vs Resistance')
plt.show()


# Set up encoding the data
X = df[['homeworld', 'unit_type']]
y = df['empire_or_resistance']

# Convert categorical features to numeric using pd.get_dummies
X_encoded = pd.get_dummies(X)

# Create a prediction model using sklearn.tree.DecisionTreeClassifier that predicts if a character is joining either the Empire or the Resistance based on their homeworld and unit_type
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier()

#Fit the model using the training data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate the accuracy of the prediction
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Code to get feature importance based of off example code given
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
feature_importances = feature_importances.sort_values('Importance', ascending=True)

# Creates a bar plot that shows feature importance
plt.figure(figsize=(8, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.show()


# Create pickle file with trained model
'''
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

'''
    