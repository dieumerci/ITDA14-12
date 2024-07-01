import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
csv_file_path = 'heart.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Assign column names
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = column_names

# Convert columns to appropriate data types if necessary
df = df.apply(pd.to_numeric, errors='ignore')

# Drop rows with missing values
df = df.dropna()

# Split the data into features and target
X = df.drop('target', axis=1)
y = df['target']

# Define categorical and numeric features
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing pipeline
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save the preprocessor for future use
import joblib
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Data preprocessing completed.")