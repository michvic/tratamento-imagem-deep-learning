## crie uma pasta com o nome 'base', na raiz do projeto onde esse escript vai ser executado
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.applications.vgg19 import VGG19
from keras.models import Model
import keras as keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

ext_img = '.jpg'
num_px = 100
num_py = 100
base_name = './base-testeConvNetSBrT2' # main folder
num_classes = 20        # number of classes
total_imgs = 20817      # total of images considering all classes

ROOT_PATH = os.path.abspath(base_name)#retorna o caminho completo da pasta ./base
dir = os.listdir(ROOT_PATH)# lista todos arquivos/pastas do diretorio ./base
dir.sort()

labels_dic = {}

labels = np.zeros((total_imgs, 1))
data = np.zeros((total_imgs, num_py, num_px, 3))

count = 0
countImg = 0

for pasta in dir:
    if os.path.isdir(ROOT_PATH + '/' + pasta):
        labels_dic[count] = pasta # cria os labels relacionando um dicetorio com um número

        imgs = glob.glob(ROOT_PATH + '/' + pasta + '/*' + ext_img)
        #imgs.sort()
        for pathImg in imgs:
            image = np.array(ndimage.imread(pathImg, flatten=False))
            #converter image para (90,100)
            data[countImg, :, :, :] = image
            labels[countImg, :] = count
            countImg += 1
        count += 1

        print('#images converted:', countImg)


print('#labels:', count)

#Train Test Split
x_train = data.reshape(data.shape[0], -1)
print("Shape of data_flatten: ", data.shape)
print("Shape of labels: ", labels.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, labels.astype('uint8'), test_size=0.4, random_state=0)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.3, random_state=2)

# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(num_py, num_px, 3))
x_test = x_test.reshape(x_test.shape[0], *(num_py, num_px, 3))
x_val = x_val.reshape(x_val.shape[0], *(num_py, num_px, 3))

print("Train shape" + str(x_train.shape))
print("Val shape" + str(x_val.shape))
print("Test shape" + str(x_test.shape))

# Set the CNN model
input_shape = (num_py, num_px, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

# Define the optimizer
# Compile the model
opt = keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])#optimizer='rmsprop'

batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        #rotation_range=20,
        #width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        #height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
#train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, save_to_dir='preview', save_format='jpeg')
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(x_val, y_val, batch_size=batch_size)

#callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='acc', factor=0.1, patience=2, min_lr=0.00001)

history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=200,
        validation_data=validation_generator,
        validation_steps=50,
        #callbacks=callbacks_list
        callbacks=[reduce_lr]
)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'], loc='upper right')
plt.title('Learning curve for the training')

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['acc','val_acc'], loc='upper right')
plt.title('Accuracy for the training')

X_train = x_train/255.
X_val = x_val/255.
X_test = x_test/255.

scores = model.evaluate(X_train, y_train, batch_size=1, verbose=0)
print("\n%s: %.2f%% (Train)" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(X_val, y_val, batch_size=1, verbose=0)
print("\n%s: %.2f%% (Val)" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(X_test, y_test, batch_size=1, verbose=0)
print("\n%s: %.2f%% (Test)" % (model.metrics_names[1], scores[1]*100))

pred_test = np.argmax(model.predict(X_test, batch_size=1, verbose=0), axis=1)
true_test = np.argmax(y_test, axis=1)
cf = confusion_matrix(true_test, pred_test)
print(cf)
print(sum(true_test==3))

for cl in range(num_classes):
    print("\n%s: %.2f%%" % (labels_dic[cl], cf[cl,cl]/sum(true_test==cl)*100))

print(classification_report(true_test, pred_test, target_names=labels_dic.values()))

plt.show()