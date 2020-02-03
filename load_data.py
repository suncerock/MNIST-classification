import csv
import numpy as np

def load_data():
    train_path = r'dataset/train.csv'
    test_path = r'dataset/test.csv'
    answer_path = r'dataset/sample_submission.csv'
    
    with open(train_path) as f:
        train = np.array(list(csv.reader(f))[1:]).astype('int')
    num_train = int(train.shape[0] * 0.9)
    num_val = train.shape[0] - num_train
    X_train = train[:num_train, 1:].reshape((num_train, 28, 28))
    y_train = train[:num_train, 0]
    X_val = train[num_train:, 1:].reshape((num_val, 28, 28))
    y_val = train[num_train:, 0]
    
    with open(test_path) as f:
        test = np.array(list(csv.reader(f))[1:]).astype('int')
    num_test = test.shape[0]
    X_test = test.reshape((num_test, 28, 28))
    
    return X_train, y_train, X_val, y_val, X_test