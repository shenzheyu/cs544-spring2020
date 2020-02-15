import json
import sys
import os
import random
import math


def file2data(file_path):
    """
    Load document to bag of words
    :param
        file_path: path to document
    :return:
        a dictionary as bag of words
    """
    data = {}
    with open(file_path, 'r', encoding='latin1') as f:
        for line in iter(f):
            tokens = line.lower().strip('\n').split(' ')
            for token in tokens:
                if token not in data.keys():
                    data[token] = 1
                else:
                    data[token] += 1
    return data


def write_output(labels):
    """
    Write the labels and path to nboutput.txt
    :param
        labels: a dictionary containing labels and path
    """
    with open('./nboutput.txt', 'w', encoding='latin1') as f:
        for path, label in labels.items():
            f.write(label + '\t' + path + '\n')


def predict(parameter, datas):
    """
    Caculate the probability of P(document | spam) and P(document | ham) to predict
    label for each document
    :param
        parameter: a dictionary containing parameters for NB model
        datas: documents waiting for predict
    :return:
        a dictionary contains labels and file path. For example,
        {'spam': './Spam or Ham/dev/4/spam/4108.2005-07-15.SA_and_HP.spam.txt',
         'ham': './Spam or Ham/dev/4/ham/0669.2000-04-10.beck.ham.txt'}
    """
    labels = {}
    for path, data in datas.items():
        p_spam_data = math.log(parameter['p_spam'])
        p_ham_data = math.log(parameter['p_ham'])
        for token in data.keys():
            if 'p_' + token + '_spam' in parameter.keys():
                p_spam_data += math.log(parameter['p_' + token + '_spam']) * data[token]
            if 'p_' + token + '_ham' in parameter.keys():
                p_ham_data += math.log(parameter['p_' + token + '_ham']) * data[token]
        if p_ham_data > p_spam_data:
            labels[path] = 'ham'
        elif p_ham_data < p_spam_data:
            labels[path] = 'spam'
        else:
            if random.random() > 0.5:
                labels[path] = 'ham'
            else:
                labels[path] = 'spam'
    return labels


def read_data(dir):
    """
    Read data from files
    :param
        dir: root direction of data
    :return:
        a dictionary contain file path and bag of word. For example:
        {"./dev/4/ham/0001.2000-01-17.beck.ham.txt": {'apple': 1, 'peer': 4},
         "./dev/4/ham/0002.2001-01-17.farmer.spam.txt": {'tree': 5, 'farm': 1}}
    """
    datas = {}
    for root, _, files in os.walk(dir):
        for file in files:
            if not file.endswith('.txt'):
                continue
            file_path = root + os.sep + file
            datas[file_path] = file2data(file_path)
    return datas


def load_model():
    """
    Load model file nbmodel.txt to dictionary
    :return:
        a dictionary contain P(spam) and P(ham) as well as conditional probabilities
        P(token|spam) and P(token|ham). For example,
        {'spam': P(spam),
         'ham': P(ham),
         'token|spam': P(token|spam),
         'token|ham': P(token|ham)}
    """
    with open('./nbmodel.txt', 'r', encoding='latin1') as f:
        parameter = json.load(f)
    return parameter


def main():
    if len(sys.argv) != 2:
        print('Please enter the right directory.')
        return
    data_dir = sys.argv[1]
    parameter = load_model()
    datas = read_data(data_dir)
    labels = predict(parameter, datas)
    write_output(labels)


if __name__ == '__main__':
    main()
