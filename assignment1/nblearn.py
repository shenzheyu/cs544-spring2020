import sys
import os
import json


def files2dict(files, dict):
    """
    Renew the word of bag dictionary from files
    :param
        files: list of file path
        dict: dictionary of word of bag
    """
    for file in files:
        f = open(file, 'r', encoding='latin1')
        for line in iter(f):
            tokens = line.lower().strip('\n').split(' ')
            for token in tokens:
                if token not in dict.keys():
                    dict[token] = 1
                else:
                    dict[token] += 1
        f.close()


def read_data(dir):
    """
    Load the train data. The directory is like
    --train
     |--1
       |--ham
         |--0001.txt
         |--0002.txt
       |--spam
         |--0001.txt
         |--0002.txt
     |--2
       |--ham
         |--0001.txt
         |--0002.txt
       |--spam
         |--0001.txt
         |--0002.txt
    :param
        dir: path to train folder
    :return:
        a dictionary contain data read from train folder, including total num of ham
        docs, total num of spam docs, two bag of word for ham and spam.
    """
    num_ham = 0
    num_spam = 0
    ham = {}
    spam = {}
    # get the relative path without the hidden files
    folders = [dir + '/' + i for i in os.listdir(dir) if not i.startswith('.')]
    for folder in folders:
        # read ham files
        ham_dir = folder + '/ham'
        ham_files = [ham_dir + '/' + i for i in os.listdir(ham_dir) if not i.startswith('.')]
        num_ham += len(ham_files)
        files2dict(ham_files, ham)
        # read spam files
        spam_dir = folder + '/spam'
        spam_files = [spam_dir + '/' + i for i in os.listdir(spam_dir) if not i.startswith('.')]
        num_spam += len(spam_files)
        files2dict(spam_files, spam)
    parameters = {'num_ham': num_ham,
                  'num_spam': num_spam,
                  'ham': ham,
                  'spam': spam}
    return parameters


def learn_model(data):
    """
    Estimate P(spam) and P(ham) as well as conditional probabilities P(token|spam) and P(token|ham)
    :param
        data: a dict contain spam data and ham data. For example,
        {'spam': {'token1': 1, 'token2': 3, 'token3': 2},
         'ham': {'token4': 1, 'token5': 3, 'token6': 2}}
    :return:
        a dict contain P(spam) and P(ham) and all conditional probabilities. For example,
        {'spam': P(spam),
         'ham': P(ham),
         'token|spam': P(token|spam),
         'token|ham': P(token|ham)}
    """
    num_spam = data['num_spam']
    num_ham = data['num_ham']
    spam = data['spam']
    ham = data['ham']
    probability = {}

    # P(spam) = count(spam) / m
    # P(ham) = count(ham) / m
    total_file = num_spam + num_ham
    p_spam = num_spam / total_file
    p_ham = num_ham / total_file
    probability['p_spam'] = p_spam
    probability['p_ham'] = p_ham

    # P(x_i|c) = (count(x_i, c) + 1) / (\sum_j count(x_j, c) + V)
    # using add-one smoothing
    vocabulary = set(spam.keys()) & set(ham.keys())
    v_size = len(vocabulary)
    sum_spam = sum(spam.values())
    sum_ham = sum(ham.values())
    for token in vocabulary:
        if token not in spam.keys():
            probability['p_' + token + '_spam'] = 1 / (sum_spam + v_size)
        else:
            probability['p_' + token + '_spam'] = (spam[token] + 1) / (sum_spam + v_size)
        if token not in ham.keys():
            probability['p_' + token + '_ham'] = 1 / (sum_ham + v_size)
        else:
            probability['p_' + token + '_ham'] = (ham[token] + 1) / (sum_ham + v_size)
    # for token, num in spam.items():
    #     probability['p_' + token + '_spam'] = num / sum_spam
    # for token, num in ham.items():
    #     probability['p_' + token + '_ham'] = num / sum_ham

    return probability


def store_model(probability):
    """
    Store the probabilities in model file nbmodel.txt with json format
    :param
        probability: dictionary of all parameters for NB model.
        {'h_spam': H(spam),
         'h_ham': H(ham),
         'h_token_spam': H(token|spam),
         'h_token_ham': H(token|ham)}
    """
    js_obj = json.dumps(probability)
    file_object = open('nbmodel.txt', 'w', encoding='latin1')
    file_object.write(js_obj)
    file_object.close()


def main():
    if len(sys.argv) != 2:
        print('Please enter the right directory.')
        return
    input_dir = sys.argv[1]
    # input_dir = './Spam or Ham/train'
    train_data = read_data(input_dir)
    probability = learn_model(train_data)
    store_model(probability)


if __name__ == '__main__':
    main()
