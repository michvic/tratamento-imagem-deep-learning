## crie uma pasta com o nome 'base', na raiz do projeto onde esse escript vai ser executado
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
import glob
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.models import Sequential

ext_img = '.jpg'
num_px = 90
num_py = 100
base_name = './base-3' # main folder
num_classes = 6        # number of classes
total_imgs = 3457      # total of images considering all classes

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
        labels_dic[pasta] = count # cria os labels relacionando um dicetorio com um número

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

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=2)

# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
x_train = x_train.reshape(x_train.shape[0], *(num_py, num_px, 3))
x_test = x_test.reshape(x_test.shape[0], *(num_py, num_px, 3))
x_val = x_val.reshape(x_val.shape[0], *(num_py, num_px, 3))

print("Train shape" + str(x_train.shape))
print("Val shape" + str(x_val.shape))
print("Test shape" + str(x_test.shape))

# Set the CNN model
input_shape = (num_py, num_px, 3)

base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)#weights=None
model = base_model.output
model = Flatten()(model)
model = Dense(128, activation='relu')(model)
model = Dropout(0.5)(model)
predictions = Dense(num_classes, activation='softmax')(model)
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
# quando usar imagenet
for layer in base_model.layers:
#for layer in base_model.layers[:15]:#fine-tunning one more convlayer
    layer.trainable = False
model.summary()

# Define the optimizer
# Compile the model
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])#optimizer='rmsprop'

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.2,
        zoom_range=0.2
)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(x_val, y_val, batch_size=batch_size)

#callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0)]
#callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)]

history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        #callbacks=callbacks_list
        #callbacks=[learning_rate_reduction]
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
print(confusion_matrix(true_test, pred_test))
plt.show()