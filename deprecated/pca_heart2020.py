import pandas as pd
import os
import numpy as np

#Definition Categorien mit niedriger Relevanz
irrelevant_attributes= ['Asthma', 'KidneyDisease', 'SkinCancer', 'Race', 'MentalHealth']

data  = pd.read_csv("DSA_Project/resources/data_clean/heart_2020_clean.csv")
relevant_attributes  = [cat for cat in data.columns if cat not in irrelevant_attributes]

numerical_attributes = [cat for cat in relevant_attributes if data[cat].dtypes != 'object']

numerical_features = data[numerical_attributes]

# Standardize values
numerical_mean = numerical_features.mean()
numerical_std = numerical_features.std()
numerical_features_standardized = (numerical_features - numerical_mean) / numerical_std

#Calculate covariance matrix
cov_matrix = numerical_features_standardized.cov()

#Calculate eigenvector and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#Sort eigenvalues by relevance
eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

#Reduce dimensions
k = 2  # number of principal components
selected_eigenvectors = np.array([pair[1] for pair in eig_pairs[:k]])

#transform values
pca_features = np.dot(numerical_features_standardized, selected_eigenvectors.T)

#join values
non_numeric_data = data[[cat for cat in relevant_attributes if cat not in numerical_features]]
reduced_data = pd.concat([non_numeric_data, pd.DataFrame(pca_features, columns=[f'pca_component{i}' for i in range(k)])], axis=1)

print(reduced_data)