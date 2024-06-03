import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Laden der Daten
file_path = 'DSA_Project/resources/data_clean/heart_2020_clean.csv'
data = pd.read_csv(file_path)
sns.set(style="whitegrid")

# Define a function to perform univariate analysis
def univariate_analysis(df):
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        if df[column].dtype == 'object':
            sns.countplot(y=column, data=df, order = df[column].value_counts().index, palette='viridis')
            plt.title(f'Verteilung von {column}')
            plt.xlabel('Anzahl')
            plt.ylabel(column)
        else:
            sns.histplot(df[column], kde=True, color='blue', binwidth=0.6)
            plt.title(f'Verteilung von {column}')
            plt.xlabel(column)
            plt.ylabel('Dichte')
        plt.show()

# Perform univariate analysis
univariate_analysis(data)

# Bivariate Analyse
def bivariate_analysis(data):
    # Korrelation zwischen numerischen Variablen
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title('Korrelation zwischen numerischen Variablen')
    plt.show()

    # Zusammenhang zwischen Zielvariable und anderen Merkmalen
    for column in data.columns:
        if data[column].dtype == 'object':
            sns.countplot(x=column, hue='HeartDisease', data=data)
            plt.title(f'Herzkrankheiten nach {column}')
            plt.show()


