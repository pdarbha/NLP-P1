import math
import time

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
    
    t = time.time()
    uni_test, bi_test = generate_un_and_big(corpus)
    print('time for file to be parsed: ' + str(time.time() - t))

    add = 0
    x = time.time()
    for bigram in bi_test:
        add += -math.log(prob_interp(uni, bi, bigram, smooth, l1, l2))
    print('len of bitest: ' + str(len(bi_test)))
    perp = math.exp(add / len(bi_test.keys()))
    print('time for loop: ' + str(time.time() - x))
    return perp


decUn, decBig = generate_un_and_big('deceptive.txt')
trUn, trBig = generate_un_and_big('truthful.txt')
print(sorted(decUn.items(), key = lambda x: x[1])[-10:])
print(sorted(decBig.items(), key = lambda x: x[1])[-10:])

print(sorted(trUn.items(), key = lambda x: x[1])[-10:])
print(sorted(trBig.items(), key = lambda x: x[1])[-10:])

print(prob_unigram(decUn, 'the'))
print(prob_bigram(decUn, decBig, 'and I'))

#print(perplexity('../validation/deceptive.txt', decUn, decBig, 1))
#print(perplexity('../validation/deceptive.txt', trUn, trBig, 1))

print(prob_unigram(decUn, 'I'))
print(prob_interp(decUn, decBig, 'and I', 0, 0.5, 0.5))
