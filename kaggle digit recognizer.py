# importing all of the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(123)  # for reproducibility
from keras.models import Sequential #this is a linear stack of the nn layers
from keras.layers import Dense, Dropout, Activation, Flatten #importing the core layers used in most nn's 
from keras.layers import Convolution2D, MaxPooling2D #importing the conv layers
from keras.utils import np_utils #we will import the utilities to help us transform data

# Let's load the training and testing data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = [train,test]

# make our rows numpy arrays so we can resize em later
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values
y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits
X_test = test.values.astype('float32')

# Since we only have one colour lets make the number of channels one 
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

# Lets standardize our data
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10)

# We can build our model now
model = Sequential()

# Now lets declare the input layer
# We have 32 filters of size 3x3, and each input is 28x28 with 1 channel
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))

# Now we can add more layers to the model very easily
# The dropout layer is to regularize our model in order to prevent overfitting
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# So far we have 2 conv layers
# Now lets add a fully connected layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          batch_size=32, nb_epoch=10, verbose=1)

# Lets make a new dataframe where we predict our x_test
predictions = model.predict_classes(X_test, verbose=0)

# Lets make a new dataframe then convert it to csv
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})

submissions.to_csv("submission.csv", index=False, header=True)