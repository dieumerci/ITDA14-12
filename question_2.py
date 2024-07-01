# QUESTION 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_file_path = 'eart.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Assign column names
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df.columns = column_names

# Cleaning
df = df.apply(pd.to_numeric, errors='ignore')

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Fill missing values or drop rows/columns with missing values if necessary
df = df.dropna()

# Plotting distributions for categorical variables
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 20))

for ax, var in zip(axes.flatten(), categorical_vars):
    sns.countplot(data=df, x=var, hue='target', ax=ax)
    ax.set_title(f'Distribution of {var} based on target')

plt.tight_layout()
plt.show()

# Plotting distributions for numeric variables
numeric_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 20))

for ax, var in zip(axes.flatten(), numeric_vars):
    sns.histplot(data=df, x=var, hue='target', multiple='stack', ax=ax)
    ax.set_title(f'Distribution of {var} based on target')

plt.tight_layout()
plt.show()

# Observations
observations = {
    "sex": "The target variable distribution shows a higher prevalence of one of the sexes in the dataset.",
    "cp": "Certain types of chest pain (cp) are more associated with the target variable.",
    "fbs": "Fasting blood sugar (fbs) does not show a significant variation based on the target variable.",
    "restecg": "Resting electrocardiographic results (restecg) show some variation with the target variable.",
    "exang": "Exercise induced angina (exang) is more prevalent in one of the target groups.",
    "slope": "The slope of the peak exercise ST segment (slope) varies with the target variable.",
    "ca": "Number of major vessels (ca) colored by fluoroscopy shows a clear pattern with the target variable.",
    "thal": "Thalassemia (thal) shows significant variation based on the target variable.",
    "age": "Age distribution shows that heart disease is more prevalent in certain age groups.",
    "trestbps": "Resting blood pressure (trestbps) varies but doesn't show a clear pattern with the target variable.",
    "chol": "Cholesterol levels (chol) vary across the target variable, with certain levels being more common.",
    "thalach": "Maximum heart rate achieved (thalach) shows a clear variation with the target variable.",
    "oldpeak": "ST depression induced by exercise relative to rest (oldpeak) shows a significant variation with the target variable."
}

for var, observation in observations.items():
    print(f"{var.capitalize()}: {observation}")


