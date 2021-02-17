import dependencies.utils as utils
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

def preprocessing(file):
    df = utils.file2df(file)
    # create sequences
    Xs, ys = utils.create_sequence(
    df[['acc_z', 'acc_XY', 'gy_x','gy_y', 'gy_z']],
    df.Activity,
    TIME_STEPS=1,
    STEP=10)
    
    # hot encoding
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    enc = enc.fit(ys)
    ys = enc.transform(ys)
    
    # split
    X_train, X_test0, y_train, y_test0 = train_test_split()
    X_test, X_val, y_test, y_val =  train_test_split()
    # Save X_train, X_test, X_val

    # scale
    scaler = RobustScaler()
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, X_val, y_train, y_test, y_val

# def model_train
# guardar los gráficos de ROC en carpeta output con la fecha --> guardar aquí tb output

# def model_test


if name == '__main__':
    #preprocessing
    X_train, X_test, X_val, y_train, y_test, y_val = preprocessing(file)
    #model train

    #model test