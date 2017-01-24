# -*- coding: utf-8 -*-

import csv
from matplotlib import pyplot as plt
import numpy as np
import scipy
from os import path
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Wrapper to load data from multiple directories, stack them together and
# return the images from center camera with corresponding steering angles
def load_data():
    img_center, steering = load_data_from_dir('data/data_1')
    
    img_center_tmp, steering_tmp = load_data_from_dir('data/data_2')
    img_center = np.vstack((img_center, img_center_tmp))
    steering = np.vstack((steering, steering_tmp))
    
    img_center_tmp, steering_tmp = load_data_from_dir('data/data_3')
    img_center = np.vstack((img_center, img_center_tmp))
    steering = np.vstack((steering, steering_tmp))
    
    img_center_tmp, steering_tmp = load_data_from_dir('data/data_4')
    img_center = np.vstack((img_center, img_center_tmp))
    steering = np.vstack((steering, steering_tmp))
    
    steering = steering.reshape(-1)
    return img_center, steering

# Load data from directories that have
#   a subdirectory called "IMG" that includes the images
#   a csv file named driving_log.csv that provides a list of images and associated steering angles
def load_data_from_dir(data_path):
    # Available fields in the csv files: 'throttle', 'brake', 'speed', 'steering', 'right', 'left', 'center'
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
        img = preprocessing(img)
        img_center.append(img)
    img_center = np.array(img_center)
    steering = np.array(steering)
    steering = steering.reshape(-1,1)
    return img_center, steering

# Image preprocessing:
#   Downsample images
#   Crop 30% of the image from top (sky area)
def preprocessing(img):
    HEIGHT = 14
    WIDTH = 40
    h,w,c = img.shape
    # Crop the top 30% of the image that only shows the sky
    img = img[int(h*0.3):,:,:]
    img = cv2.resize(img,(WIDTH,HEIGHT))
    return img

#### Data Augmentation:
#    Keras has a super helpful data generator function called `ImageDataGenerator`
#    that can augment the available data with
#    new images synthesized by applying multiple wellknown transformations on them.
#    After trying all transformation with different purterbation values the following parameters turned out to work best with
#    the current architecture and dataset:
#    * rotation_range: 5 degrees (Degree range for random rotations)
#    * width_shift_range: 5% (Range for random horizontal shifts)
#    * height_shift_range: 5% (Range for random vertical shifts)
#    * zoom_range: 5% (Range for random zoom in/out)

def data_augmentation(x, y, batch_size):
    from keras.preprocessing.image import ImageDataGenerator
    import keras.backend as K

    option = dict(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range= 5,
    width_shift_range= 0.05,
    height_shift_range= 0.05,
    shear_range= 0.,
    zoom_range= 0.05,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    dim_ordering=K.image_dim_ordering())

    datagen = ImageDataGenerator(**option)
    datagen.fit(x)
    for X_batch, Y_batch in datagen.flow(x, y, batch_size=batch_size):
        X_batch, Y_batch = horizontal_flip(X_batch, Y_batch)
        yield X_batch, Y_batch

#Horizontal Flip:
#    Image outputs from the Keras generator are flipped horizontally with 50% randomness to increase the number of samples. 
#    Once an image is fliped, the associated steering angle is multipled by -1 to be consistent with the new image.
def horizontal_flip(x,y):
    x_ = np.copy(x)
    y_ = np.copy(y)
    for i in range(x.shape[0]):
        if (np.random.rand(1)>0.5):
            x_[i] = np.fliplr(x_[i])
            y_[i] = -np.copy(y_[i])
    return x_, y_
        
# Reducing zero-angle samples
#    The time series and histogram of the steering angles in training data is shown in the figure below. 
#    Since most of the track is straight, there is a large population around zero angle. 
def reduce_zero_angle_samples(x_train_all, y_train_all):
    zero_steering = y_train_all == 0
    idx = np.where(zero_steering)[0]
    np.random.shuffle(idx)
    idx = idx[:int(idx.shape[0]*0.01)]
    zero_steering[idx] = False
    
    x_train_all = x_train_all[np.logical_not(zero_steering)]
    y_train_all = y_train_all[np.logical_not(zero_steering)]
    return x_train_all, y_train_all

# Splitting Data Between Training and Validation Sets:
#    Since the consecutive samples are very similar to each other shuffling the data and spliting it into training and validation cause
#    the training set to have images almost identical to the ones in the validation set. To keep the validation set different from the
#    training set the samples are first divided into batches of 10 consecutive samples and then 33% of the batches are randomly selected as
#    the validation set. The histogram of training and validation sets are shown below. Note that the distributions are similar and there is no 
#    large peak at 0 anymore.
def split_data_test_validation(x_train_all, y_train_all):
    x_train_all, y_train_all = reduce_zero_angle_samples(x_train_all, y_train_all)
    n_frames = x_train_all.shape[0]
    
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
    return x_train, y_train, x_valid, y_valid

def visualize_results(steering, y_train, y_valid):
    # Plot the time series of steering angles
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(steering)
    plt.title("Steering angles (all data)")
    plt.ylabel("Normalized angle")
    plt.xlabel("Sample")
    
    # Plot histogram of steering angles
    plt.subplot(1,2,2)
    plt.hist(steering,100)
    plt.title("Steering angles histogram (all data)")
    plt.ylabel("Counts")
    plt.xlabel("Angle")
    plt.draw()
    plt.savefig('./Figures/steering_original')
    
    plt.figure(figsize=(12,6))
    # Plot histogram of angles in training set
    plt.subplot(1,2,1)
    plt.hist(y_train,100)
    plt.title("Angles after reducing zero values (train)")
    plt.ylabel("Counts")
    plt.xlabel("Angle")
    
    # Plot histogram of angles in training set
    plt.subplot(1,2,2)
    plt.hist(y_valid,100)
    plt.title("Angles after reducing zero values (train)")
    plt.ylabel("Counts")
    plt.xlabel("Angle")
    plt.draw()
    plt.savefig('./Figures/steering_equalized')
    
def shahnet(input_shape):
    keep_prob = 0.5 
    activation = 'elu'
    
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    
    model.add(Convolution2D(8, 3, 3, subsample=(1, 1), border_mode="valid", activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    
    model.add(Convolution2D(16, 1, 1, subsample=(1, 1), border_mode="valid", activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    ## === Fully connected layers
    model.add(Flatten())
    model.add(Dense(50, activation=activation))
    model.add(Dropout(keep_prob))
    model.add(Dense(10, activation=activation))
    model.add(Dropout(keep_prob))
    model.add(Dense(1, activation='tanh'))
    
    #optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-10)
    optimizer = 'adam'
    model.compile(loss='mse',optimizer=optimizer)    
    
    # Save the best model by validation mean squared error
    checkpoint = ModelCheckpoint("model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # Reduce learning rate when the validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6)    
    # Stop training when there is no improvment. 
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')
    callbacks=[checkpoint, early_stop, reduce_lr]
    
    # Model visualization
    from keras.utils.visualize_util import plot
    plot(model, to_file='./Figures/model.png',show_shapes=True)
    model.summary()
    return model, callbacks


def shahnet_small(input_shape):
    keep_prob = 0.5 
    activation = 'elu'
    
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(8, 1, 1, subsample=(1, 1), border_mode="valid", activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    
    model.add(Convolution2D(8, 1, 1, subsample=(1, 1), border_mode="valid", activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    
    model.add(Convolution2D(8, 1, 1, subsample=(1, 1), border_mode="valid", activation=activation))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid', dim_ordering='default'))
    ## === Fully connected layers
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    model.add(Dense(1, activation='tanh'))
    
    #optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-10)
    optimizer = 'adam'
    model.compile(loss='mse',optimizer=optimizer)    
    
    # Save the best model by validation mean squared error
    checkpoint = ModelCheckpoint("model_small.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # Reduce learning rate when the validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6)    
    # Stop training when there is no improvment. 
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=1, mode='min')
    callbacks=[checkpoint, early_stop, reduce_lr]
    
    
    # Model visualization
    from keras.utils.visualize_util import plot
    plot(model, to_file='./Figures/model_small.png',show_shapes=True)
    model.summary()
    return model, callbacks

#model, callbacks = shahnet_small(input_shape=(HEIGHT, WIDTH, CHANNEL))


    
# %% MAIN FUNCTION

    
def main(model_size):
    import keras.backend as K
    HEIGHT = 14
    WIDTH = 40
    CHANNEL = 3
    
    
    print("==================")
    print("Loading data ...")
    img_center, steering = load_data()
    x_train_all = img_center
    y_train_all = steering
    
    x_train, y_train, x_valid, y_valid = split_data_test_validation(x_train_all, y_train_all)
    
    
    visualize_results(steering, y_train, y_valid)
    plt.savefig('1.png')
    
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
    
    
    
        
    
    #%%
    
    if (model_size.lower()=="small"):
        model, callbacks = shahnet_small(input_shape=(HEIGHT, WIDTH, CHANNEL))
        model_name = "model_small"
    elif(model_size.lower()=="large"):
        model, callbacks = shahnet(input_shape=(HEIGHT, WIDTH, CHANNEL))
        model_name = "model"
    else:
        raise Exception('Unknown model size. Use either "small" or "large".')
    
    
    # Save the model architecture
    with open(model_name+'.json','w') as f:
        f.write(model.to_json())
    
    # Train the model by using Keras' generator
    history = model.fit_generator(data_augmentation(x_train, y_train, batch_size=64),
                                      validation_data=(x_valid, y_valid),
                                      samples_per_epoch=len(x_train), nb_epoch=200,callbacks=callbacks)
    
    # Plot validation and training loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    conv1 = K.function(model.inputs, [model.layers[1].output]) 
    conv2 = K.function(model.inputs, [model.layers[3].output]) 
    conv3 = K.function(model.inputs, [model.layers[5].output]) 
    
    img = x_train[0]
    img = img.reshape(np.hstack((1,img.shape)))
    a = np.array(conv1([img]))
    for i in range(a.shape[4]):
        plt.imshow(np.squeeze(a[:,:,:,:,i]),cmap='gray')
        plt.show()
    plt.show()
#%%

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a deep CNN network for autonomous driving.')
    parser.add_argument('--model_size', type=str, default="large",
    help='use large for a 6-layer CNN (+4K parameters) and small for a 4-layer CNN (217 parameters).')
    args = parser.parse_args()
    main(args.model_size)



