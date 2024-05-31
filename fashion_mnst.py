from  tensorflow.keras.datasets import fashion_mnist
from  tensorflow.keras.models  import Sequential
from  tensorflow.keras.layers   import Dense
from  tensorflow.keras.utils   import to_categorical

( X_train_data, Y_train_data ), ( X_test_data, Y_test_data ) = fashion_mnist.load_data()

X_train = X_train_data.reshape( Y_train_data.shape[0], 28 * 28 ).astype( 'float32' ) / 255 
X_test  = X_test_data.reshape( Y_test_data.shape[0], 28 * 28 ).astype( 'float32' ) / 255 

OneHot_train = to_categorical( Y_train_data, 10)
OneHot_test  = to_categorical( Y_test_data, 10)

my_model = Sequential()

my_model.add( Dense( units=64 , input_shape=( 28 * 28 ,) , activation='relu' ) )
my_model.add( Dense( units=512, activation='relu' ) )
my_model.add( Dense( units=64 , activation='relu' ) )
my_model.add( Dense( units=10 , activation='softmax' ) )

my_model.compile( optimizer= 'rmsprop', loss='categorical_crossentropy', metrics=['accuracy'] )

my_model.fit( X_train , OneHot_train , batch_size = 128 , epochs = 20 , validation_split=0.2  )


score = my_model.evaluate( X_test , OneHot_test , verbose=0 )
print( 'accuracy:{}'.format( score[1] ))

