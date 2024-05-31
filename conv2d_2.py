from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.layers import MaxPooling2D , AveragePooling2D
from tensorflow.keras.utils  import to_categorical

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = mnist.load_data()

X_train = X_train_data.reshape( Y_train_data.shape[0], 28 , 28 , 1 ).astype( 'float32' ) / 255 
X_test  = X_test_data.reshape( Y_test_data.shape[0], 28 , 28 , 1 ).astype( 'float32' ) / 255 

OneHot_train = to_categorical( Y_train_data, 10)
OneHot_test  = to_categorical( Y_test_data, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu' , padding='same' , use_bias=False ))

model.add(AveragePooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu' , strides=(3, 3) ))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()

# model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )
# model.fit( X_train , OneHot_train , batch_size = 64 , epochs = 5 , validation_split=0.2  )

# test_loss , test_acc = model.evaluate( X_test , OneHot_test , verbose=0 )
# print( 'accuracy:{}'.format( test_acc ))

