from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Convolution2D as Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.utils  import to_categorical

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = fashion_mnist.load_data()

X_train = X_train_data.reshape( Y_train_data.shape[0], 28 , 28 , 1 ).astype( 'float32' ) / 255 
X_test  = X_test_data.reshape( Y_test_data.shape[0], 28 , 28 , 1 ).astype( 'float32' ) / 255 

OneHot_train = to_categorical( Y_train_data, 10)
OneHot_test  = to_categorical( Y_test_data, 10)

model = Sequential()
model.add( Conv2D( filters=32, kernel_size=(3, 3), input_shape=(28,28,1), activation='relu'))

model.add( MaxPooling2D(pool_size=(2, 2)))

model.add( Conv2D( filters=64, kernel_size=(3, 3), activation='relu'))

model.add( MaxPooling2D(pool_size=(2, 2)))

model.add( Conv2D( 64, (3, 3), activation='relu'))

model.add( Flatten() )

model.add( Dense( 64, activation='relu'))

model.add( Dense( 10, activation='softmax'))

model.summary()

model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )
model.fit( X_train , OneHot_train , batch_size = 64 , epochs = 5 , validation_data=( X_test , OneHot_test ) )

score = model.evaluate( X_test , OneHot_test , verbose=0 )
print( 'accuracy:{}'.format( score[1] ))
