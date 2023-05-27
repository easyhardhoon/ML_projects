import tensorflow as tf
import numpy as np

#0 PREPROCESSING : make(load) data
#--------------------------------------------------------
X= []; Y = []
for i in range(6):
    lst = list(range(i,i+4))
    X.append(list(map(lambda c: [c/10], lst)))
    Y.append((i+4)/10)
X = np.array(X)
Y = np.array(Y)
for i in range(6) : print(X[i]);print(Y[i])
#--------------------------------------------------------

#------------------------------------------------------------------------------
#0_2 PREPROCESSING : make data for LSTM
X = []
Y = []
for i in range(3000):
# 0~1 사이의 랜덤한 숫자 100 개를 만듭니다.
    lst = np.random.rand(100)
# 마킹할 숫자 2개의 인덱스를 뽑습니다.
    idx = np.random.choice(100, 2, replace=False)
# 마킹 인덱스가 저장된 원-핫 인코딩 벡터를 만듭니다.
    zeros = np.zeros(100)
    zeros[idx] = 1
# 마킹 인덱스와 랜덤한 숫자를 합쳐서 X 에 저장합니다.
    X.append(np.array(list(zip(zeros, lst))))
# 마킹 인덱스가 1인 값들만 서로 곱해서 Y 에 저장합니다.
    Y.append(np.prod(lst[idx]))
print(X[0], Y[0])
#------------------------------------------------------------------------------


#1. SimpleRNN
#--------------------------------------------------------
model = tf.keras.Sequential(
        #input shape : timesteps, input_dim
        #default activation func : tanh. 
        [tf.keras.layers.SimpleRNN(units=10, return_sequences=False, input_shape=[4,1]), 
        #defaut activation func is linearistor . (y==x)
        tf.keras.layers.Dense(units=1)])
#--------------------------------------------------------


#2. SimpleRNN + return_sequences = True
#--------------------------------------------------------
model = tf.keras.Sequential(
        #input shape : timesteps, input_dim
        [tf.keras.layers.SimpleRNN(units=30, return_sequences=True, input_shape=[100,2]), 
        tf.keras.layers.SimpleRNN(units=30), 
        #defaut activation func is linear . (y==x)
        tf.keras.layers.Dense(units=1)])        
#--------------------------------------------------------


#3. LSTM
#--------------------------------------------------------
model = tf.keras.Sequential(
        #input shape : timesteps, input_dim
        [tf.keras.layers.LSTM(units=30, return_sequences=True, input_shape=[100,2]), 
        tf.keras.layers.LSTM(units=30), 
        #defaut activation func is linear . (y==x)
        tf.keras.layers.Dense(units=1)])
#--------------------------------------------------------



#4. GRU
#--------------------------------------------------------
# lower parameter set size . 20% down than LSTM
model = tf.keras.Sequential(
        #input shape : timesteps, input_dim
        [tf.keras.layers.GRU(units=30, return_sequences=True, input_shape=[100,2]), 
        tf.keras.layers.GRU(units=30), 
        #defaut activation func is linear . (y==x)   #GRU has pros in low parameter size
        tf.keras.layers.Dense(units=1)])
#--------------------------------------------------------
#simple
#model.compile(optimizer='adam', loss='mse')
#model.summary()
#model.fit(X,Y, epochs=100, verbose=0)
#print(model.predict(X))
#----------------------------------------------------
#LSTM
X = np.array(X)
Y = np.array(Y)
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(X[:2560], Y[:2560], epochs=100, validation_split=0.2)
model.evaluate(X[2560:], Y[2560:])
prediction = model.predict(X[2560:2560+5])
for i in range(5):
    print(Y[2560+i], '\t', prediction[i][0], '\tdiff:', abs(prediction[i][0] - Y[2560+i]))
prediction = model.predict(X[2560:])
cnt = 0
for i in range(len(prediction)):
    if abs(prediction[i][0] - Y[2560+i]) > 0.04:
        cnt += 1
print('correctness:', (440 - cnt) / 440 * 100, '%')
