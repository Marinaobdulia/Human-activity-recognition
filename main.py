import dependencies.utils as utils
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras
import sys

def preprocessing(file):
    df = utils.file2df(file)
    # create sequences
    Xs, ys = utils.create_sequence(
    df[['acc_z', 'acc_XY', 'gy_x','gy_y', 'gy_z']],
    df.Activity,
    time_steps=1,
    step=10)
    
    # hot encoding
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(ys)
    ys = enc.transform(ys)
    
    # split
    X_train, X_test0, y_train, y_test0 = train_test_split(Xs, ys, test_size = 0.4)
    X_test, X_val, y_test, y_val =  train_test_split(X_test0, y_test0, test_size = 0.5)
    
    # Save X_train, X_test, X_val
    np.save('./Output/X_train.npy', X_train)
    np.save('./Output/y_train.npy', y_train)

    np.save('./Output/X_test.npy', X_test)
    np.save('./Output/y_test.npy', y_test)

    np.save('./Output/X_val.npy', X_val)
    np.save('./Output/y_val.npy', y_val)

    # scale
    #scaler = RobustScaler()
    #scaler = scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

    return X_train, X_test, X_val, y_train, y_test, y_val

def model_train(X_train, y_train, graphing = 'Yes'):
    model = keras.Sequential()
    model.add(
        keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=128,
            input_shape=[X_train.shape[1], X_train.shape[2]]
        )
        )
    )
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=128, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['acc'])

    history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
    )

    if graphing == 'Yes':
        graph = utils.grapher(history)

    return model


def model_test(model, X_test, y_test):
    return model.evaluate(X_test, y_test)


if __name__ == '__main__':
    file = 'HAR_database.mat'
    # file = sys.arg[1]
    #preprocessing
    X_train, X_test, X_val, y_train, y_test, y_val = preprocessing(file)
    
    #model train
    model = model_train(X_train, y_train)
    
    #model test
    loss, acc = model_test(model, X_test, y_test)

    #model save
    filename = './Output/finalized_model.h5'
    model.save(filename)