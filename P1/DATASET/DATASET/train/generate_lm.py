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




decUn, decBig = generate_un_and_big('deceptive.txt')
trUn, trBig = generate_un_and_big('truthful.txt')
print(sorted(decUn.items(), key = lambda x: x[1])[-10:])
print(sorted(decBig.items(), key = lambda x: x[1])[-10:])

print(sorted(trUn.items(), key = lambda x: x[1])[-10:])
print(sorted(trBig.items(), key = lambda x: x[1])[-10:])

print(prob_unigram(decUn, 'the'))
print(prob_bigram(decUn, decBig, 'and I'))