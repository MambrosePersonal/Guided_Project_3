import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Clean Data using replace and ffill
'''
df = pd.read_csv("troop_movements10m.csv")
df['unit_type'] = df['unit_type'].replace('invalid_unit', 'unknown')
df['location_x'] = df['location_x'].ffill()
df['location_y'] = df['location_y'].ffill()
df.to_parquet('troop_movements10m.parquet')
'''

# Pull pickle trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

#Read in data from troop_movements10m.parquet file
data = pd.read_parquet('troop_movements10m.parquet')

# Set columns and endcode data
X = data[['homeworld', 'unit_type']]
X_encoded = pd.get_dummies(X)

# Add new column with predictions
predictions = model.predict(X_encoded) 
data['predictions'] = predictions
print(data)