from  tensorflow.keras.datasets import cifar10
from  tensorflow.keras.models  import Sequential
from  tensorflow.keras.layers   import Dense, Flatten, Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization
from  tensorflow.keras.utils   import to_categorical
import time

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = cifar10.load_data()

print('X_train_data.shape :', X_train_data.shape)
print('X_test_data.shape  :', X_test_data.shape)
print('Y_train_data.shape :', Y_train_data.shape)
print('Y_test_data.shape  :', Y_test_data.shape)

X_train = X_train_data.astype("float32") / 255.
X_test  = X_test_data.astype("float32") / 255.

OneHot_train = to_categorical( Y_train_data, 10)
OneHot_test  = to_categorical( Y_test_data, 10)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',         input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense( 10 ))
model.add(Activation('softmax'))

import tensorflow.keras.callbacks as callbacks
earlyStopping = callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )
t = time.time()
model.fit( X_train , OneHot_train , batch_size = 128 , epochs = 100 , validation_split=0.2, callbacks=[earlyStopping] )
print("model.fit 用了 : {:.3f} 秒".format(time.time() - t))

score = model.evaluate( X_test , OneHot_test , verbose=0 )
print( 'accuracy:{}'.format( score[1] ))

model.summary()

# cifar10 所有圖片分為 10 個類別
#    0: airplain, 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
exit()
import numpy as np
import  matplotlib.pyplot as plt
plt.imshow(X_test_data[33], cmap='binary')
plt.show()

plt.imshow(np.hstack(X_test_data[30:40]))
plt.show()

plt.imshow( np.vstack([ np.hstack(X_test_data[i:i+10]) for i in range(0 , 100 , 10) ]) )
plt.show()


