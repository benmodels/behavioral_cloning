# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv
from matplotlib import pyplot as plt
import numpy as np
import scipy
from os import path
from tqdm import tqdm
import cv2

HEIGHT = 20
WIDTH = 40
CHANNEL = 3

#%% === Load data
def load_data(data_path):
    # dict_keys(['throttle', 'brake', 'speed', 'steering', 'right', 'left', 'center'])
    steering = []
    center = []
    with open(data_path+'/driving_log.csv') as f:
        log_dict = csv.DictReader(f)
        for line in log_dict:
            steering.append(float(line['steering']))
            center.append(line['center'])
    # Load images
    img_center = []
    for path_c in tqdm(center):
        img_name = path.split(path_c)[-1]
        p = path.join(data_path, 'IMG', img_name)
        img = scipy.misc.imread(p)
        img = cv2.resize(img,(WIDTH,HEIGHT)) #/255.0 * 2.0 - 1.0
        img_center.append(img)
    img_center = np.array(img_center)
    steering = np.array(steering)
    steering = steering.reshape(-1,1)
    return img_center, steering


img_center, steering = load_data('data')

img_center_tmp, steering_tmp = load_data('data_2')
img_center = np.vstack((img_center, img_center_tmp))
steering = np.vstack((steering, steering_tmp))

img_center_tmp, steering_tmp = load_data('data_3')
img_center = np.vstack((img_center, img_center_tmp))
steering = np.vstack((steering, steering_tmp))

img_center_tmp, steering_tmp = load_data('data_4')
img_center = np.vstack((img_center, img_center_tmp))
steering = np.vstack((steering, steering_tmp))

del img_center_tmp, steering_tmp

steering = steering.reshape(-1)
n_frames = len(steering)
        

        


# Plot the time series of steering angles
plt.plot(steering)
plt.show()

# Plot histogram of steering angles
plt.hist(steering,100)
plt.show()

# Plot one image for negative, zero and positive angles
neg = np.where(steering < -0.6)[0][0]
pos = np.where(steering >  0.6)[0][0]
zer = np.where(np.logical_and(steering > -0.02, steering < 0.02))[0][200]

plt.figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1,3,1)
plt.imshow(img_center[neg])
plt.axis('off')
plt.title('Negative angle')

plt.subplot(1,3,2)
plt.imshow(img_center[zer])
plt.axis('off')
plt.title('Zero angle')

plt.subplot(1,3,3)
plt.imshow(img_center[pos])
plt.axis('off')
plt.title('Positive angle')


#%% Train/Test/Validation data


x_train_all = img_center
y_train_all = steering

plt.hist(y_train_all,100)
plt.show()

if (True):
    zero_steering = y_train_all == 0
    idx = np.where(zero_steering)[0]
    np.random.shuffle(idx)
    idx = idx[:int(idx.shape[0]*0.25)]
    zero_steering[idx] = False
    
    x_train_all = x_train_all[np.logical_not(zero_steering)]
    y_train_all = y_train_all[np.logical_not(zero_steering)]
    n_frames = x_train_all.shape[0]

plt.hist(y_train_all,100)
plt.show()
#x_train = np.copy(x_train_all[:1000])
#y_train = np.copy(y_train_all[:1000])

# Divide the train set into chunks of seq_len frames
from sklearn.model_selection import train_test_split
seq_len = 10 
n_chunkes = int(n_frames/seq_len)
i_chunk_train,i_chunk_valid,_,_ = train_test_split(np.arange(n_chunkes, dtype=np.int32).reshape(-1,1),
                                                   np.arange(n_chunkes, dtype=np.int32).reshape(-1,1), test_size=0.33,random_state = 1)
assert i_chunk_train.shape[0]+i_chunk_valid.shape[0] == n_chunkes

i_valid = np.zeros((n_frames), dtype=np.bool)
for i in i_chunk_valid:
    i_valid[i[0]*seq_len:(i[0]+1)*seq_len] = True
x_valid, y_valid = x_train_all[i_valid], y_train_all[i_valid]
x_train, y_train = x_train_all[np.logical_not(i_valid)], y_train_all[np.logical_not(i_valid)]
assert x_valid.shape[0]+x_train.shape[0] == n_frames



#%%
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

option = dict(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range= 0.0,
    width_shift_range= 0.0,
    height_shift_range= 0.0,
    shear_range= 0.,
    zoom_range= 0,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())

datagen = ImageDataGenerator(**option)
datagen.fit(x_train)

def horizontal_flip(x,y):
    x_ = np.copy(x)
    y_ = np.copy(y)
    for i in range(x.shape[0]):
        if (np.random.rand(1)>0.5):
            x_[i] = np.fliplr(x_[i])
            y_[i] = -np.copy(y_[i])
    return x_, y_

#i = 0

#for X_batch, Y_batch in datagen.flow(x_train, y_train, batch_size=32):
#    X_batch, Y_batch = horizontal_flip(X_batch, Y_batch)
#    loss = model.train_on_batch(X_batch, Y_batch)
#    print (loss)

plt.subplot(1,2,1)
plt.hist(y_train,20)
plt.subplot(1,2,2)
plt.hist(y_valid,20)
plt.show()


def datagen_aug(x, y, batch_size):
    for X_batch, Y_batch in datagen.flow(x, y, batch_size=batch_size):
        X_batch, Y_batch = horizontal_flip(X_batch, Y_batch)
        yield X_batch, Y_batch

#for X_batch, Y_batch in datagen_aug(x, y, batch_size=32):
#    loss = model.train_on_batch(X_batch, Y_batch)
#    print (loss)
    

#%%



keep_prob = 0.5 
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
activation = 'elu'
input_shape = (HEIGHT, WIDTH, CHANNEL)
# Layer 1
#model.add(Convolution2D(10, 5, 5, input_shape=(HEIGHT,  WIDTH, CHANNEL), subsample=(2,2), border_mode='valid'))
#model.add(Activation(activation))
model = Sequential()

model.add(Lambda(lambda x: x / 255.0 - 0.5,
                             input_shape=input_shape,
                             output_shape=input_shape))
model.add(Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
model.add(Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
model.add(Convolution2D(16, 1, 1, subsample=(1, 1), border_mode="valid", activation=activation))
model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
#model.add(Convolution2D(16, 1, 1, subsample=(1, 1), border_mode="valid", activation=activation))
#model.add(Dropout(keep_prob))

# Layer 2
#model.add(Convolution2D(10, 5, 5, subsample=(2,2), border_mode='valid'))
#convout1 = Activation(activation) 
#model.add(convout1)
#
#convout1_f = K.function(model.inputs, [convout1.output])


# Layer 3
#model.add(Convolution2D(6, 5, 5, subsample=(2,2), border_mode='valid'))
#model.add(Activation(activation))
## Layer 4
#model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid'))
#model.add(Activation(activation))
## Layer 5
#model.add(Convolution2D(64, 3, 3, subsample=(1,1), border_mode='valid'))
#model.add(Activation(activation))
#
## === Fully connected layers
model.add(Flatten())
## Layer FC1
#model.add(Dense(1164))
#model.add(Activation(activation))
#model.add(Dropout(keep_prob))
# Layer FC2
#model.add(Dense(100))
#model.add(Activation(activation))
#model.add(Dropout(keep_prob))
# Layer FC3
model.add(Dense(50))
model.add(Activation(activation))
model.add(Dropout(keep_prob))
# Layer FC4
model.add(Dense(10))
model.add(Activation(activation))
model.add(Dropout(keep_prob))
# Layer Fc5
model.add(Dense(1))
model.add(Activation('tanh'))


#model.compile(loss='mse',optimizer='adam',metrics=['mse'])    
#optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-10)
optimizer = 'adam'
model.compile(loss='mse',optimizer=optimizer)    

# Save the best model by validation mean squared error
checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)    
# Stop training when there is no improvment. 
# This is to speed up training, the accuracy is not affected, because the checkpoint will pick-up the best model anyway
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
#%%
#for i in range(10):
#    history = model.fit(x_train_all, y_train_all, batch_size=64, nb_epoch=10, validation_data=(x_train_all, y_train_all))
#    loss, mse  = model.evaluate(x_train_all, y_train_all, batch_size=64)
#    print(mse)
#,callbacks=[checkpoint, early_stop]

model_json = model.to_json()
model_name = 'model.json'
with open(model_name,'w') as f:
    f.write(model_json)


history = model.fit_generator(datagen_aug(x_train, y_train, batch_size=64),
                                  validation_data=(x_valid, y_valid),
                                  samples_per_epoch=len(x_train), nb_epoch=100,callbacks=[checkpoint, early_stop, reduce_lr])
#for i in range(20):
#    history = model.fit_generator(datagen_aug(x_train, y_train, batch_size=64),
#                                  validation_data=(x_valid, y_valid),
#                                  samples_per_epoch=len(x_train), nb_epoch=100)
#    model_json = model.to_json()
#    model_name = 'model'+str(i+1)+'.json'
#    with open(model_name,'w') as f:
#        f.write(model_json)
#    model.save('model'+str(i+1)+'.h5')
#    
#    lr = K.get_value(model.optimizer.lr)
#    lr*= 0.977
#    K.set_value(model.optimizer.lr, lr)
#    print('**** Model saved to: '+model_name+'\n')
    


#K.set_value(model.optimizer.lr, 0.0005)
#history = model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_data=(x_valid, y_valid))
#history = model.fit(x_train_all, y_train_all, batch_size=64, nb_epoch=3, validation_data=(x_train_all, y_train_all))
#history = model.fit(x_train, y_train, batch_size=64, nb_epoch=10, validation_data=(x_valid, y_valid))
#     summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

a = np.array(convout1_f([x_train[:2]]))
for i in range(6):
    plt.imshow(np.squeeze(a[0,1,:,:,i]))
    plt.show()
#%%





