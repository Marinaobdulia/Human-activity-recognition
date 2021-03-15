# meter todas las funciones xra importarlas
from scipy.io import loadmat, savemat
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

def file2df(file):
    data = loadmat(file)
    n_people = len(data['database_training'])
    
    for i in range(n_people):
        if i==0:
            df = pd.DataFrame(data['database_training'][i][0].transpose(), 
            columns=['acc_z', 'acc_XY', 'gy_x', 'gy_y', 'gy_z'])
            df['User']=len(data['database_training'][i][1][0])*[0]
            df['Activity']=data['database_training'][i][1][0]
            df['Timestamp']=range(len(data['database_training'][i][1][0]))
        else:
            df2 = pd.DataFrame(data['database_training'][i][0].transpose(), 
            columns=['acc_z', 'acc_XY', 'gy_x','gy_y', 'gy_z'])
            df2['User']=len(data['database_training'][i][1][0])*[i]
            df2['Activity']=data['database_training'][i][1][0]
            df2['Timestamp']=range(len(data['database_training'][i][1][0]))
            df = df.append(df2, ignore_index=True)
    return df

def create_sequence(X, y, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)

def grapher(history, output):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output+'/model_accuracy.png')
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(output+'/model_loss.png')
    plt.close()

