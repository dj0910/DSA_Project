import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten laden
data = pd.read_csv("DSA_Project/resources/data_dirty/heart_predictions.csv").dropna(how='any')

# Convert 'thal' and 'ca' columns to numeric, forcing errors to NaN
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')

# Drop rows with NaN values in 'thal' and 'ca' columns
data_cleaned = data.dropna(subset=['thal', 'ca'])

# Streudiagramme erstellen
sns.pairplot(data_cleaned, vars=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'], hue='target', diag_kind='kde')
plt.suptitle('Streudiagramme für ausgewählte numerische Variablen', y=1.02)
plt.show()