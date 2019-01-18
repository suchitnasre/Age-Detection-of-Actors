import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Convolution2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam 
from keras.models import model_from_json
from scipy.misc import imread, imshow, imresize, imsave
from keras import regularizers


# Importing the Dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

Train_X = []
for i in train.ID:
    img_name = imread('X/'+i)
    img = imresize(img_name, size = (32, 32))
    img = img.astype('float32')
    Train_X.append(img)    
    print(len(Train_X))

Train_X = np.stack(Train_X)


# Loading the images automatically from Test folder
Test_X = []
for i in test.ID:
    img_name_2 = imread('Y/'+i)
    img_2 = imresize(img_name_2, size = (32, 32))
    img_2 = img_2.astype('float32')
    Test_X.append(img_2)
    print(len(Test_X))

Test_X = np.stack(Test_X)
    
# Feature scaling
Train_X /= 255
Test_X /= 255



# Fit the convolutional model
input_shape = (32, 32, 3)
batch = 25
epoch = 25
optimizer = Adam(lr = 0.0001)
N_classes = len(train.Class.value_counts())


model = Sequential()
model.add(Convolution2D(32, kernel_size=(3,3), input_shape = input_shape, padding = 'same', kernel_regularizer= regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size = (2,2)))
    
model.add(Convolution2D(64, kernel_size=(3,3), padding = 'same', kernel_regularizer= regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Convolution2D(128, kernel_size=(3,3), padding = 'same', kernel_regularizer= regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size = (2,2)))

model.add(Convolution2D(256, kernel_size=(3,3), padding = 'same', kernel_regularizer = regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(MaxPool2D(pool_size = (2,2)))
      
    
model.add(Flatten())
    
model.add(Dense(512, kernel_regularizer= regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))   
model.add(Dense(N_classes))
model.add(Activation('softmax'))


# Compile the model
model.compile(optimizer = optimizer, 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])


# Augment the Data
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Train',
        target_size=(32, 32),
        batch_size=batch,
        class_mode='categorical')

testing_set = test_datagen.flow_from_directory(
        'Test',
        target_size=(32, 32),
        batch_size=batch,
        class_mode='categorical')

history = model.fit_generator(training_set,
                         steps_per_epoch=18406/batch,
                         epochs=epoch,
                         validation_data=testing_set,
                         validation_steps=1500/batch)


# Serialize model to json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Serialize weights to HDF5
model.save_weights("AgeIdentification.hdf5")


# Load json and create model
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weight into new model
loaded_model.load_weights("AgeIdentification.hdf5")


# Summarize the history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize the history for Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Predict the Test_y
Test_y = model.predict_classes(Test_X)
training_set.class_indices


# Save the Test_y file and submit it!!
test['Class'] = Test_y
test.to_csv("trial.csv", index = False)
testing = pd.read_csv("trial.csv")
testing = testing.replace(0, "MIDDLE")
testing = testing.replace(1, "OLD")
testing = testing.replace(2, "YOUNG")

testing.to_csv("Submission_4.csv", index = False)