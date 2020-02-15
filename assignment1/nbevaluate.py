import sys


def evaluate(output, goal):
    correct = 0
    classify = 0
    belong = 0
    for path, label in output.items():
        if path.find(goal) != -1:
            belong += 1
            if label == goal:
                correct += 1
        if label == goal:
            classify += 1
    if classify != 0:
        precision = correct / classify
    else:
        precision = 0
    if belong != 0:
        recall = correct / belong
    else:
        recall = 0
    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    print(goal + " precision: " + str(precision))
    print(goal + " recall: " + str(recall))
    print(goal + " f1: " + str(f1))


def load_output(file_name):
    output = {}
    with open(file_name, 'r', encoding='latin1') as f:
        for line in iter(f):
            label = line[0: line.find('\t')]
            path = line[line.find('\t'): -1]
            output[path] = label
    return output


def main():
    if len(sys.argv) != 2:
        print('Please enter the nboutput file name.')
        return
    file_name = sys.argv[1]
    output = load_output(file_name)
    evaluate(output, 'spam')
    evaluate(output, 'ham')


if __name__ == '__main__':
    main()