import keras
import numpy as np
import sys

def load_dataset(dataset):
    X_val = np.load(dataset[0])
    y_val = np.load(dataset[1])
    return X_val, y_val


if __name__ == '__main__':
    model_file = sys.argv[1]
    model = keras.models.load_model(model_file)

    dataset = [sys.argv[2], sys.argv[3]]
    X_val, y_val = load_dataset(dataset)

    y_val_calculated = model.predict(X_val)
    loss, acc = model.evaluate(X_val, y_val)
    # print --> log (módulo logging)
    