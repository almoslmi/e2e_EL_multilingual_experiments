from keras.models import load_model
import numpy as np
import nltk
from nltk.corpus import stopwords
import os

import candidate_generator as cg
import entvec_encoder as eenc
import laserencoder as lenc

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def feature(words, trie, kv, enc):
    sent = ' '.join(words)
    X_list = []
    cand_list = []
    gram_list = []
    ind_list = []
    mentions = {}
    for i in range(1, 6):
        for wind, gram in zip(ngram(range(len(words)), i), ngram(words, i)):
            mention = ' '.join(list(gram))
            candidates = cg.generate_with_linkprob_and_windex(
                mention, wind[0], wind[-1], trie)
            encoded = eenc.encode_with_linkprob_and_windex(candidates, kv)
            tmp_X2 = []
            tmp_X4 = []
            tmp_cand = []
            tmp_ind = []
            tmp_gram = []
            for target in encoded:
                tmp_cand.append(target["candname"])
                tmp_ind.append((target["begin"], target["end"]))
                tmp_X2.append(target["entvec"])
                tmp_X4.append([target["linkprob"]])
                tmp_gram.append(mention)
                mentions[mention] = True
            if tmp_X2:
                X_list.append([None, np.array(tmp_X2), None, np.array(tmp_X4)])
                cand_list.append(tmp_cand)
                gram_list.append(tmp_gram)
                ind_list.append(tmp_ind)
    mentions = [x.replace("\n", " ") for x, _ in mentions.items()]
    mvecs = enc.encode('\n'.join([sent] + mentions))
    x1 = mvecs[0]
    mvecs = dict(zip(mentions, mvecs[1:]))
    for i, grams in enumerate(gram_list):
        tmp_X3 = []
        tmp_X1 = []
        for mention in grams:
            tmp_X3.append(mvecs[mention.replace("\n", " ")])
            tmp_X1.append(x1)
        X_list[i][0] = np.array(tmp_X1)
        X_list[i][2] = np.array(tmp_X3)
    return X_list, cand_list, gram_list, ind_list


def predict(model, X_list, cand_list, gram_list, ind_list, threshold=0.5):
    out = []
    for X, cands, grams, wind in zip(X_list, cand_list, gram_list, ind_list):
        y_preds = [x[0] for x in model.predict(X)]
        if np.max(y_preds) < threshold:
            continue
        index = np.argmax(y_preds)
        out.append({
            "entity": cands[index],
            "mention": grams[index],
            "word_index": wind[index]
        })
    return out


if __name__ == "__main__":
    trie = cg.load("../data/mention_stat.marisa")
    kv = eenc.load("../data/enwiki_20180420_100d.bin")
    enc = lenc.Encoder()
    model = load_model("../data/model_wiki_tmp.h5")
    sent = None
    while sent != "-1":
        threshold = float(input("threshold>"))
        sent = input("sent>")
        words = nltk.word_tokenize(sent)
        X_list, cand_list, gram_list, ind_list = feature(words, trie, kv, enc)
        result = predict(model, X_list, cand_list, gram_list, ind_list,
                         threshold)
        print(result)
