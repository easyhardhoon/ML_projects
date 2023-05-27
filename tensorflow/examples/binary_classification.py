import pandas as pd
red = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
white = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', sep=';')
print(red.head())
print(white.head)
red['type'] = 0  # append data
white['type'] = 1 # append data
print(red.head(2)) # print head above 2 lines
print(white.head(2))  # print head above 2 lines

wine = pd.concat([red, white])  #concate.concatenate. merge
print(wine)
print(wine.describe()) # summary all's data based statici


import matplotlib.pyplot as plt
plt.hist(wine['type'])
plt.xticks([0,1])
plt.show()
print(wine['type'].value_counts())
wine_norm = (wine - wine.min()) / (wine.max() -wine.min()) #normalization
print(wine_norm.head())
print(wine_norm.describe())


wine_shuffle = wine_norm.sample(frac=1)
print(wine_shuffle.head())
print(type(wine_shuffle))
wine_np = wine_shuffle.to_numpy()  # move data type : numpy
print(type(wine_np))  # pandas.core.frame.Dataframe -> numpy.ndarray 
print(wine_np[:5])

import tensorflow as tf
train_idx = int(len(wine_np) * 0.8)
train_X, train_Y = wine_np[:train_idx, :-1], wine_np[:train_idx, -1]
test_X, test_Y = wine_np[train_idx:, :-1], wine_np[train_idx:, -1]
print(train_X[0])
print(train_Y[0])
print(test_X[0])
print(test_Y[0])
train_Y = tf.keras.utils.to_categorical(train_Y, num_classes=2)
test_Y = tf.keras.utils.to_categorical(test_Y, num_classes=2)
print(train_Y[0])
print(test_Y[0])


#data preprocessing
#------------------------------------------------------------------------
#dnn model predicting

model = tf.keras.Sequential([
tf.keras.layers.Dense(units=48, activation='relu', input_shape=(12,)),
tf.keras.layers.Dense(units=24, activation='relu'),
tf.keras.layers.Dense(units=12, activation='relu'),
tf.keras.layers.Dense(units=2, activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.07), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
