import os
import sys

import nltk
import numpy as np
from keras.models import load_model
from nltk.corpus import stopwords

import candidate_generator as cg
import entvec_encoder as eenc
import laserencoder as lenc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

stps = stopwords.words("english")


def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def feature(words, trie, kv, enc):
    x1 = enc.encode(' '.join(words))[0]
    X_list = []
    cand_list = []
    gram_list = []
    for i in range(1, 6):
        for gram in ngram(words, i):
            stps_flag = False
            for g in list(gram):
                if g in stps:
                    stps_flag = True
                    break
            if stps_flag:
                continue
            mention = ' '.join(list(gram))
            candidates = [x for x, _ in cg.generate(mention, trie)]
            encoded = eenc.encode(candidates, kv)
            tmp_X2 = []
            tmp_X1 = []
            tmp_cand = []
            tmp_gram = []
            for target in encoded:
                tmp_cand.append(target["candname"])
                tmp_X2.append(target["entvec"])
                tmp_X1.append(x1)
                tmp_gram.append(mention)
            if tmp_X1:
                X_list.append([np.array(tmp_X1), np.array(tmp_X2)])
                cand_list.append(tmp_cand)
                gram_list.append(tmp_gram)
    return X_list, cand_list, gram_list


def predict(model, X_list, cand_list, gram_list):
    out = []
    for X, cands, grams in zip(X_list, cand_list, gram_list):
        y_preds = [x[0] for x in model.predict(X)]
        if np.max(y_preds) < 0.3:
            continue
        index = np.argmax(y_preds)
        out.append(cands[index])
    return list(set(out))


if __name__ == "__main__":
    sent = sys.argv[1]
    words = nltk.word_tokenize(sent)
    trie = cg.load("../data/mention_stat.marisa")
    kv = eenc.load("../entity_vector/enwiki_20180420_100d.bin")
    enc = lenc.Encoder()
    model = load_model("../model_best.h5")
    X_list, cand_list, gram_list = feature(words, trie, kv, enc)
    result = predict(model, X_list, cand_list, gram_list)
    print(result)
