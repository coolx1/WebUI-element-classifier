from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.models import model_from_json

classifier = Sequential()
classifier.add(Conv2D(36, (3, 3), input_shape = (50, 50, 3),strides=1, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(72, (3, 3),strides = 2, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units = 11, activation = 'softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2)
training_set = train_datagen.flow_from_directory('training_set',batch_size = 32,target_size=(50,50))

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = train_datagen.flow_from_directory('test_set',batch_size = 10,target_size=(50,50))
classifier.fit_generator(training_set,epochs = 54,validation_data = test_set,validation_steps = 56)


# print("Saving Classifier...")
# pickle_out = open("classifier.pickle","wb")
# pickle.dump(classifier, pickle_out)
# pickle_out.close()
# print("Classifier saved!")

model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")


# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")