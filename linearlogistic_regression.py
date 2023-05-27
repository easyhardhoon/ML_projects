import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Fit a neural network to the Default data. 
# have a look at  Labs 10.9.1 - 10.9.2 for guidance.
# Use a single hidden layer with 10 units, and dropout regularization. 
# Compare the classficiation performance of your model with that of linear logistic regression

#1. data preprocessing
data = pd.read_csv('Default.csv', dtype={'balance':float, 'income':float})
data['default'] = data['default'].map({'No': 0, 'Yes': 1})
data['student'] = data['student'].map({'No': 0, 'Yes': 1})
X = data.drop('default', axis=1)
y = data['default']

#2. data split for train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#3. model make
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#4. model training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

model2 = LinearRegression()
model2.fit(X,y)
predictions2 = model2.predict(X)
predictions = model.predict(X_test)
print("prediction by custom model : ",predictions)
print("prediction by LinearRegression : ",predictions2 )

#5. model evaluate
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy (Custom Model):', accuracy)
accuracy2 = model2.score(X, y)
print('Accuracy (Linear Regression):', accuracy2)


# SUMMARY
# 1. binary classfication : custom model (similar to linear logistic regression) has high performance
# 2. Linear Regression model is acutually fit to "regression" , not "binary classification"
# 3. Finally, custom model based on linear logistic regression is more efficient than linear-regression model
