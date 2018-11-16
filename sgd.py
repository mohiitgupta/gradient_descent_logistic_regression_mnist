import numpy as np
import sys
from util import *
from graph_plotting import *

def train_sgd(train_data, num_epochs, learning_rate, lamda, test_data):
    data_size = len(train_data)
    feature_length = train_data.shape[1]
    weights = np.random.uniform(-0.1,0.1,[10,feature_length])
    bias = np.ones((data_size,1))
    prev_loss = 1000
    batch_size_loss = 1000
    epoch_axis = []
    train_accuracy_axis = []
    test_accuracy_axis = []
    epoch=0
    while epoch < num_epochs:
        loss = np.zeros(10)
        np.random.shuffle(train_data)
        examples, labels, classifier_labels = get_features_labels(train_data, bias)
        for i,example in enumerate(examples):
            delta_weights = np.zeros((10,feature_length))
            z = np.sum(weights*example, axis = 1)
            y_pred = sigmoid(z)
            delta_weights += learning_rate*np.outer(y_pred-classifier_labels[:,i],example)
            loss+=classifier_labels[:,i]*np.log(y_pred+0.0000000001)+(1-classifier_labels[:,i])*np.log(1-y_pred+0.0000000001)
            weights = weights - delta_weights - learning_rate*lamda*weights
            if i%batch_size_loss==0:
                loss = loss*-1.0/batch_size_loss + lamda*np.sum(np.square(weights), axis = 1)/2.0
                loss = np.sum(loss)/10.0
                
                train_prediction, train_labels, train_accuracy = inference(train_data[:data_size], weights)
                test_prediction, test_labels, test_accuracy = inference(test_data, weights)
                '''
                 for graph plotting
                '''
                epoch_axis.append(epoch)
                train_accuracy_axis.append(train_accuracy)
                test_accuracy_axis.append(test_accuracy)
                
                print "epoch ", epoch, ": Training loss: ", loss, " Training Accuracy: ", train_accuracy, " Test Accuracy: ", test_accuracy
                delta_loss = prev_loss - loss
                prev_loss = loss
                loss = np.zeros(10)
                epoch += 1
                # print "delta loss ", delta_loss
                if delta_loss > 0 and delta_loss < 0.0002 and epoch > 200:
                    return weights, epoch_axis, train_accuracy_axis, test_accuracy_axis
    return weights, epoch_axis, train_accuracy_axis, test_accuracy_axis

def main(argv):
    if len(argv) < 3:
        print "Usage: python sgd.py [regularization? True/False] [Feature_Type? type1/type2] [path to DATA_FOLDER]"
    else:
        feature_type = argv[1]
        path = argv[2]
        train_data_size = 10000
        epochs = 300
        learning_rate = 0.001
        if argv[0] == 'True':
            lamda = 0.0001
        else:
            lamda = 0
        
        # print "lamda is ", lamda, " feature_type is ", feature_type

        train_data = preprocess(path + '/train-images-idx3-ubyte.gz', path + '/train-labels-idx1-ubyte.gz', feature_type)
        test_data = preprocess(path + '/t10k-images-idx3-ubyte.gz', path + '/t10k-labels-idx1-ubyte.gz', feature_type)

        gd_weights, epoch_axis, train_accuracy_axis, test_accuracy_axis = train_sgd(train_data[:10000], epochs, learning_rate, lamda, test_data)
        # test_prediction, test_labels, test_accuracy = inference(test_data_type1, gd_weights)

        plot_learning_curves('Epoch', epoch_axis, train_accuracy_axis, test_accuracy_axis, 'convergence')


if __name__ == '__main__':
    main(sys.argv[1:])