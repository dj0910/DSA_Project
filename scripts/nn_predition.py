import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

print(os.getcwd())
data = pd.read_csv("DSA_Project/resources/data_dirty/heart_predictions.csv").dropna(how='any')

col_label = 'target'

col_features = [feat for feat in data.columns if feat != col_label]

labels = data[col_label]
features = data[col_features]

rand_state  = 100
max_iter = 1000000
hidden_layers = [14,7]
features_train, features_test, label_train, label_test = train_test_split(features, labels, random_state=rand_state, test_size=0.3, shuffle=True)
# Define Models

##  Linear Model
### Logistic Regression
regression_lin = LogisticRegression(max_iter=max_iter, random_state=rand_state)


## Neural Networks

### Classification (MLP)
classification_nn = MLPClassifier(
    hidden_layer_sizes=hidden_layers,
    activation='logistic',
    solver='adam',
    max_iter=max_iter,
    random_state=rand_state
)

### Regressr (MLP)
regression_nn = MLPRegressor(
    hidden_layer_sizes=hidden_layers,
    activation='logistic',
    solver='adam',
    max_iter=max_iter,
    random_state=rand_state
)

# Train model
regression_lin.fit(features_train, label_train)
classification_nn.fit(features_train, label_train)
regression_nn.fit(features_train, label_train)

# Test models

pred_lin_lr  = regression_lin.predict(features_test)
pred_class_nn = classification_nn.predict(features_test)
pred_reg_nn = regression_nn.predict(features_test)


# Print results
mse = mean_squared_error(label_test, pred_class_nn)
print("MSE Logistic Regression", mean_squared_error(label_test, pred_lin_lr))
print("MSE Classification NN", mean_squared_error(label_test, pred_class_nn))
print("MSE Regression NN", mean_squared_error(label_test, pred_reg_nn))

