import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Pfad zur CSV-Datei und Zielspalte
csv_path = "resources/data_clean/heart_2020_clean.csv"
target_column = "HeartDisease"

# Laden des Datensatzes
df = pd.read_csv(csv_path)

# Aufteilen in Features und Ziel
X = df.drop(columns=[target_column])
y = df[target_column]

# Identifizieren der kategorialen und numerischen Spalten
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['number']).columns

# Anwenden von One-Hot-Encoding auf die kategorialen Daten
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

X_processed = preprocessor.fit_transform(X)

# Durchführen der PCA
pca = PCA(n_components=5)
pca.fit(X_processed)

# Abrufen der PCA-Ladungen
loadings = pca.components_.T

# Erstellen eines DataFrames der Ladungen
feature_names = preprocessor.get_feature_names_out()
loadings_df = pd.DataFrame(loadings, columns=[f'PC{i+1}' for i in range(5)], index=feature_names)

# Anzeigen der wichtigsten Attribute für jede Hauptkomponente
print("Top 5 Attribute für jede Hauptkomponente:")
for i in range(5):
    top_attributes = loadings_df.iloc[:, i].abs().sort_values(ascending=False).head(5).index
    print(f"PC{i+1}: {', '.join(top_attributes)}")
