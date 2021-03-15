import keras
import numpy as np
import sys
import os.path
import argparse
import logging


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return str(arg)  # return filename string

if __name__ == '__main__':
    logging.basicConfig(filename='prediction.log', filemode='w', level=logging.INFO,
    format='%(asctime)-15s - %(levelname)s- %(name)s - %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model", required=True,
                    help="keras model", metavar="FILE",
                    type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-x", dest="x", required=True,
                help="data matrix (X)", metavar="FILE",
                type=lambda x: is_valid_file(parser, x))
    parser.add_argument("-y", dest="y", required=False,
                help="data classification", metavar="FILE",
                type=lambda x: is_valid_file(parser, x))
    args = parser.parse_args()

    model = keras.models.load_model(args.model)

    if args.y is not None:
        X_val = np.load(args.x)
        y_val = np.load(args.y)
        loss, acc = model.evaluate(X_val, y_val)
        logging.info(f'Model evaluation results: {acc*100:.2f}% acc {loss*100:.2f}% loss')
    else:
        X_pred = np.load(args.x)
        y_pred = model.predict(X_pred)
        np.save('predictions', y_pred)
        logging.info(f'Prediction results are available on this folder')




   