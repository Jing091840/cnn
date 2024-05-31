import numpy as np
import sklearn

from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(x_train_all,y_train_all),(x_test,y_test)= fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000],x_train_all[5000:]
y_valid, y_train = y_train_all[:5000],y_train_all[5000:]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform( x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scaler.transform( x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scaler.transform( x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
for _ in range(20):
    model.add(keras.layers.Dense(100, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    """
    #先進行全連接但不激活
    model.add(keras.layers.Dense(100))
    
    #進行標準化
    model.add(keras.layers.BatchNormalization())
    
    #進行激活函數處理
    model.add(keras.layers.Activation("relu"))
    """
else:
    model.add(keras.layers.Dense(10,activation="softmax"))

model.compile(loss= "sparse_categorical_crossentropy",optimizer= "sgd", metrics = ["accuracy"])
model.summary()
    
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3)
]

history = model.fit(x_train_scaled,y_train, epochs=100,validation_data=(x_valid_scaled,y_valid),callbacks = callbacks)

score = model.evaluate(x_test_scaled,y_test)
print( 'accuracy:{}'.format( score[1] ))




