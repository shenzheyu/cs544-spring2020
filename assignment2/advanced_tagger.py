import sys
import pycrfsuite

from hw2_corpus_tool import get_data


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile, 'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = []
        for val in splitLine[1:]:
            embedding.append(float(val))
        model[word] = embedding
    print("Done.", len(model), " words loaded!")
    return model


def load_corpus(dir):
    try:
        corpus = []
        generator = get_data(dir)
        while True:
            corpus.append(next(generator))
    except StopIteration:
        pass
    return corpus


def utterance2features(dialogue, i):
    features = []

    if dialogue[i].pos is not None:
        # 71.90
        features.append(str(len(dialogue[i].pos)))

        # 71.45
        # if dialogue[i].pos[-1].token == '.':
        #     features.append('END_STOP')
        # elif dialogue[i].pos[-1].token == '?':
        #     features.append('END_QUESTION')

        # 71.68
        # features.append(str(len(dialogue[i].text)))

        # 71.52
        # if dialogue[i].text[-1] == '?':
        #     features.append('END_QUESTION')

        # 71.27
        # text = dialogue[i].text
        # if text is not None:
        #     if text.find('I think') != -1 or text.find('I believe') != -1 or text.find('it seems') != -1 or text.find('I mean') != -1:
        #         features.append('OPINION')

        # 71.38
        pos = dialogue[i].pos
        for p in pos:
            # 71.17
            # stopwords = {'a-', 'the-', 'an-', 'in-', 'to-', 'a', 'the', 'an', 'in', 'to', ',', '.'}
            # if p.token.lower() not in stopwords:

            # features.append('TOKEN_' + p.token.lower())
            try:
                features.append('WORDVECTOR_' + str(model[p.token.lower()]))
            except:
                pass
            features.append('TOKEN_' + p.token.lower())
            features.append('POS_' + p.pos)
        if i == 0:
            features.append('FIRST_UTTERANCE')
        else:
            current_speaker = dialogue[i].speaker
            last_speaker = dialogue[i - 1].speaker
            if current_speaker != last_speaker:
                features.append('SPEAKER_CHANGE')

            # 71.52
            # if dialogue[i - 1].text is not None:
            #     features.append('AFTER_' + dialogue[i - 1].text)

            # 71.69
            if dialogue[i - 1].text is not None:
                if dialogue[i - 1].text.find('?') != -1:
                    features.append('AFTER QUESTION')

            # 71.58
            # if dialogue[i - 1].pos is not None:
            #     prev_pos = dialogue[i - 1].pos
            #     for p in prev_pos:
            #         features.append('PREVTOKEN_' + p.token.lower())
            #         features.append('PREVPOS_' + p.pos)
    else:
        features.append('NO_WORD')

    return features


def dialogue2features(dialogue):
    return [utterance2features(dialogue, i) for i in range(len(dialogue))]


def dialogue2labels(dialogue):
    return [utterance.act_tag for utterance in dialogue]


def crf_train(X_train, y_train):
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 100,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train('SWBD-DAMSL-advanced.crfsuite')
    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])


def crf_predict(X_test):
    tagger = pycrfsuite.Tagger()
    tagger.open('SWBD-DAMSL-advanced.crfsuite')
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    return y_pred


def print_tagger(y_pred, file_name):
    with open(file_name, 'w') as f:
        for dialogue in y_pred:
            for act in dialogue:
                print(act, file=f)
            print('', file=f)


def crf_evaluate(y_pred, y_test):
    true = 0
    all = 0
    wrong = {}
    for dialogue_pred, dialogue_test in zip(y_pred, y_test):
        for act_pred, act_test in zip(dialogue_pred, dialogue_test):
            if act_pred == act_test:
                true += 1
                all += 1
            else:
                all += 1
                if act_test not in list(wrong.keys()):
                    wrong[act_test] = 1
                else:
                    wrong[act_test] += 1
    print('accuracy: ', true / all)
    print('wrong', wrong)


def main():
    if len(sys.argv) != 4:
        print('Please enter the right parameter.')
        return
    input_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_file = sys.argv[3]
    train_corpus = load_corpus(input_dir)
    test_corpus = load_corpus(test_dir)

    global model
    model = loadGloveModel('./glove.6B.50d.txt')

    X_train = [dialogue2features(dialogue) for dialogue in train_corpus]
    y_train = [dialogue2labels(dialogue) for dialogue in train_corpus]
    crf_train(X_train, y_train)

    X_test = [dialogue2features(dialogue) for dialogue in test_corpus]
    y_test = [dialogue2labels(dialogue) for dialogue in test_corpus]
    y_pred = crf_predict(X_test)
    print_tagger(y_pred, output_file)

    crf_evaluate(y_pred, y_test)


if __name__ == '__main__':
    main()
