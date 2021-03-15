import dependencies.utils as utils
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras
import sys
from datetime import datetime
import os
import os.path
import argparse
import logging


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return str(arg)  # return filename string


def preprocessing(file, output):
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
    for data, name in [[X_train, 'X_train'], [y_train, 'y_train'], 
    [X_test, 'X_test'], [y_test, 'y_test'],
    [X_val, 'X_val'], [y_val, 'y_val']]:
        np.save(output+'/'+name, data)    
    # scale
    #scaler = RobustScaler()
    #scaler = scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)

    return X_train, X_test, X_val, y_train, y_test, y_val, Xs, ys

def model_train(X_train, y_train, output, graphing = 'Yes'):
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
        graph = utils.grapher(history, output)

    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="filename", required=True,
                    help="input file with two matrices", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()
    file = args.filename

    # en el parser se puede meter tb el nivel de logging

    output = './Output_'+str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    os.mkdir(output)
    
    logging.basicConfig(filename=output+'/main.log', filemode='w', level=logging.INFO,
    format="%(asctime)s;%(levelname)s;%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S')

    X_train, X_test, X_val, y_train, y_test, y_val, Xs, ys = preprocessing(file,output)
    logging.info('Model sucessfully preprocessed')
    model, history = model_train(X_train, y_train, output)
    loss, acc = model.evaluate(X_test, y_test)
    logging.info(f'Model evaluation results: {acc*100:.2f}% acc {loss*100:.2f}% loss')

    model, history = model_train(Xs, ys, output)
    logging.info(f'Final model sucessfully trained')

    filename = output+'/finalized_model.h5'
    model.save(filename)
    logging.info(f'Final model sucessfully saved: {filename}')
