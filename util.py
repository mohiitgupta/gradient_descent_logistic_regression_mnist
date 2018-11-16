import numpy as np
import struct
import sys
import gzip

def sigmoid(value):
    value[value > 100] = 100
    value[value < -100] = -100
    return 1/(1+np.exp(-value))

def inference(test_data, weights):
    data_size = len(test_data)
    bias = np.ones((data_size,1))
    examples, labels, classifier_labels = get_features_labels(test_data, bias)
    prediction = np.ones(data_size, dtype = int)
    correct = 0
    for i, example in enumerate(examples):
        activation_values = sigmoid(np.sum(weights*example, axis = 1))
        prediction[i] = np.argmax(activation_values)
        if prediction[i] == labels[i]:
            correct += 1
    accuracy = correct*1.0/data_size
    return prediction, labels, accuracy
    
def read_file(filename):
    with gzip.open(filename,'rb') as fp:
        zero, data_type, dims = struct.unpack('>HBB', fp.read(4))
        shape = tuple(struct.unpack('>I', fp.read(4))[0] for d in range(dims))
        np_array = np.frombuffer(fp.read(), dtype=np.uint8).reshape(shape)
    return np_array

def preprocess(image_file, label_file, feature_type):
    if feature_type == 'type1':
        return preprocess_type1(image_file, label_file)
    elif feature_type == 'type2':
        return preprocess_type2(image_file, label_file)
    else:
        print "invalid feature type"
        return

def preprocess_type1(image_file, label_file):
    images = read_file(image_file)
    labels = read_file(label_file)
    if (len(labels) > 10000):
        labels = labels[:10000]
        images = images[:10000]    
    images = images/255.0
    images = images.reshape( (10000, 784))

    labels = labels.reshape(-1,1)
    data = np.concatenate((images, labels), axis=1)
    np.random.shuffle(data)
    return data

def preprocess_type2(image_file, label_file):
    images = read_file(image_file)
    labels = read_file(label_file)
    if (len(labels) > 10000):
        labels = labels[:10000]
        images = images[:10000]    
    images = images/255.0
    images_sample = np.zeros((10000, 14, 14))
    for i,image in enumerate(images):
        for j in range(14):
            for k in range(14):
                images_sample[i][j][k] = max(image[j*2][k*2], image[j*2][k*2+1],
                                                 image[j*2+1][k*2], image[j*2+1][k*2+1])
    images_sample = images_sample.reshape( (10000, 196))

    labels = labels.reshape(-1,1)
    data = np.concatenate((images_sample, labels), axis=1)
    np.random.shuffle(data)
    return data

def get_features_labels(data, bias):
    examples = data[:,:-1]
    labels = data[:,-1]
    classifier_labels = np.zeros((10, len(labels)))
    for i in range(10):
        classifier_labels[i][labels == i] = 1
    examples = np.append(examples, bias, 1)
    return examples, labels, classifier_labels