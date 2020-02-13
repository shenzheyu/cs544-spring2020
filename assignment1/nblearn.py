import sys
import os


def files2dict(files, dict):
    for file in files:
        f = open(file, 'r', encoding='latin1')
        for line in iter(f):
            tokens = line.strip('\n').split(' ')
            for token in tokens:
                if token not in dict.keys():
                    dict[token] = 1
                else:
                    dict[token] += 1
        f.close()


def read_data(dir):
    ham = {}
    spam = {}
    folders = os.listdir(dir)
    for folder in folders:
        # read ham files
        ham_dir = folder + '/ham'
        ham_files = os.listdir(ham_dir)
        files2dict(ham_files, ham)
        # read spam files
        spam_dir = folder + '/spam'
        spam_files = os.listdir(spam_dir)
        files2dict(spam_files, spam)
    parameters = {'ham': ham,
                  'spam': spam}
    return parameters


def learn_model(data):
    # TODO estimate P(spam) and P(ham) as well as conditional probabilities P(token|spam) and P(token|ham)
    pass


def store_model(probability):
    # TODO store the probabilities in model file nbmodel.txt
    pass


def main():
    if len(sys.argv != 1):
        print('The argument is a data directory.')
    input_dir = sys.argv[0]
    # input_dir = './Spam or Ham/train'
    train_data = read_data(input_dir)
    probability = learn_model(train_data)
    store_model(probability)


if __name__ == 'main':
    main()
