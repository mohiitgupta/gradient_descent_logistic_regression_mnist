import numpy as np
import struct
# from f1_macro import calculate_macro_f1_score

def sigmoid(value):
    return 1/(1+np.exp(-value))

def read_file(filename):
    with open(filename,'rb') as fp:
        zero, data_type, dims = struct.unpack('>HBB', fp.read(4))
        shape = tuple(struct.unpack('>I', fp.read(4))[0] for d in range(dims))
        np_array = np.frombuffer(fp.read(), dtype=np.uint8).reshape(shape)
    return np_array

def preprocess(image_file, label_file):
    images = read_file(image_file)
    labels = read_file(label_file)
    if (len(labels) > 10000):
        labels = labels[:10000]
        images = images[:10000]    
    images = images/255.0
    images = images.reshape( (10000, 784))

    labels = labels.reshape(-1,1)
    data = np.concatenate((images, labels), axis=1)
#     np.random.shuffle(data)
    return data

def get_true_label(digit, perceptron_type):
    if digit == perceptron_type:
        return 1
    return -1

def get_features_labels(data, bias):
    examples = data[:,:-1]
    labels = data[:,-1]
    examples = np.append(examples, bias, 1)
    return examples, labels

def inference(test_data, weights):
    data_size = len(test_data)
    bias = np.ones((data_size,1))
    examples, labels = get_features_labels(test_data, bias)
    prediction = np.ones(data_size, dtype = int)
    correct = 0
    for i, example in enumerate(examples):
        activation_values = np.sum(weights*example, axis = 1)
        prediction[i] = np.argmax(activation_values)
        if prediction[i] == labels[i]:
            correct += 1
#     print correct*1.0/data_size*100
    return prediction, labels

# def get_f1_score(test_data, weights):
#     prediction, true_labels = inference(test_data, weights)
#     f1_score = calculate_macro_f1_score(prediction, true_labels)
#     return f1_score