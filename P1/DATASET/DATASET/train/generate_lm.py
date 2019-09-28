import math

def generate_un_and_big(text):
    un = {}
    big = {}
    with open(text) as file:
        for line in file:
            tokens = line.split(" ")
            for i, token in enumerate(tokens):
                if token in un:
                    un[token] += 1
                else:
                    un[token] = 1
                if i == 0:
                    big['<s> '+token] = 1
                if i < len(tokens) - 1:
                    s = token + " " + tokens[i+1]
                    if s in big:
                        big[s] += 1
                    else:
                        big[s] = 1
    return un, big

def generate_un_and_big_from_line(line):
    un = {}
    big = {}
    tokens = line.split(" ")
    for i, token in enumerate(tokens):
        if token in un:
            un[token] += 1
        else:
            un[token] = 1
        if i == 0:
            big['<s> '+token] = 1
        if i < len(tokens) - 1:
            s = token + " " + tokens[i+1]
            if s in big:
                big[s] += 1
            else:
                big[s] = 1
    return un, big

def prob_unigram(grams, unigram, smooth=0):
    if unigram in grams:
        return (grams[unigram] + smooth)/(sum(grams.values()) + smooth*len(grams.keys()))
    else:
        return smooth/(sum(grams.values()) + smooth*len(grams.keys()))

def prob_bigram(unigrams, bigrams, bigram, smooth=0):
    words = bigram.split(" ")
    bigprob = smooth/(sum(bigrams.values()) + smooth*len(bigrams.keys()))
    if bigram in bigrams:
        bigprob += bigrams[bigram]/(sum(bigrams.values()) + smooth*len(bigrams.keys()))
    return bigprob/prob_unigram(unigrams, words[0], smooth)

def prob_interp(unigrams, bigrams, phrase, smooth = 0, l1 = 0, l2 = 1):
    pr_u = prob_unigram(unigrams, phrase.split(" ")[1], smooth)
    pr_b = prob_bigram(unigrams, bigrams, phrase, smooth)

    return l1 * pr_u + l2 * pr_b

def perplexity(corpus, uni, bi, smooth = 0, l1 = 0, l2 = 1):
    
    bi_test = generate_un_and_big_from_line(corpus)[1]

    add = 0
    for bigram in bi_test:
        add += -math.log(prob_interp(uni, bi, bigram, smooth, l1, l2))
    perp = math.exp(add / len(bi_test.keys()))
    return perp


# decUn, decBig = generate_un_and_big('deceptive.txt')
# trUn, trBig = generate_un_and_big('truthful.txt')


import pandas as pd
# df = pd.DataFrame(columns=['smooth', 'l1', 'truth_acc', 'dec_acc'])
# #for s in range(1,4):
# for i in range(1,11):
#     # truth_acc = 0
#     # dec_acc = 0
#     smooth = float(i)/100.0
#     l1 = 1#0.95 + float(i)/100.0
#     l2 = 0#1-l1
#     row = {}
#     row['smooth'] = smooth
#     row['l1'] = l1
#     with open('../validation/truthful.txt') as file:
#         acc = 0    
#         for line in file:
#             if perplexity(line, decUn, decBig, smooth, l1, l2) > perplexity(line, trUn, trBig, smooth, l1, l2):
#                 acc+=1
#         row['truth_acc'] = float(acc)/128.0
#     with open('../validation/deceptive.txt') as file:
#         acc = 0
#         for line in file:
#             if perplexity(line, decUn, decBig, smooth, l1, l2) < perplexity(line, trUn, trBig, smooth, l1, l2):
#                 acc+=1
#         row['dec_acc'] = float(acc)/128.0
#     df = df.append(row, ignore_index=True)
#     print('smooth:{}, l1:{}, deceptive:{}%, truthful:{}%'.format(smooth, l1, row['dec_acc'], row['truth_acc']))
# df.to_csv('results.csv')
# print(perplexity('../validation/deceptive.txt', decUn, decBig, 1))
# print(perplexity('../validation/deceptive.txt', trUn, trBig, 1))
# print(perplexity('../validation/truthful.txt', decUn, decBig, 1))
# print(perplexity('../validation/truthful.txt', trUn, trBig, 1))


def generate_results(decUn, decBig, trUn, trBig):
    import pandas as pd
    df = pd.DataFrame(columns=['Id', 'Prediction'])
    with open('../test/test.txt') as file:
        for i, line in enumerate(file):
            row = {}
            row['Id'] = i
            if perplexity(line, decUn, decBig, 0.1, 0.99, 0.01) > perplexity(line, trUn, trBig, 0.1, 0.99, 0.01):
                row['Prediction'] = 0
            else:
                row['Prediction'] = 1
            df = df.append(row, ignore_index = True)
    df.to_csv('preds_2.csv')

#generate_results(decUn, decBig, trUn, trBig)

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

def generate_bow(truthful, deceptive):
    vec = DictVectorizer()
    bow = []
    labels = []
    with open(truthful) as file:
        for line in file:
            labels.append(0)
            bow.append(generate_un_and_big_from_line(line)[0])
    with open(deceptive) as file:
        for line in file:
            labels.append(1)
            bow.append(generate_un_and_big_from_line(line)[0])
    X = vec.fit_transform(bow).toarray()
    y = np.array(labels)
    v = { word:i for i, word in enumerate(vec.get_feature_names()) }
    return X, y, v

def train_model(X, y, alpha):
    clf = MultinomialNB(alpha = alpha)
    clf.fit(X, y)
    return clf

def text_lines(text):
    with open(text) as file:
        for i, line in enumerate(file):
            pass
        return i+1 

def predict_val(text, clf, feature_names):
    with open(text) as file:
        bow = np.zeros((text_lines(text), len(feature_names)))
        for i, line in enumerate(file):
            bag_of_unigrams = generate_un_and_big_from_line(line)[0]
            for word in bag_of_unigrams:
                if word in feature_names:
                    bow[i, feature_names[word]] = bag_of_unigrams[word]
        return clf.predict(bow)

X, y, v = generate_bow('truthful.txt', 'deceptive.txt')
clf = train_model(X, y, 1)
predictions = predict_val('../test/test.txt', clf, v)
pd.DataFrame(predictions).to_csv('nb_results.csv')            






    
        
            


