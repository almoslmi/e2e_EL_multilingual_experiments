from keras.models import load_model
import numpy as np
import os
import MeCab
import entvec_encoder as eenc
import laserencoder as lenc
import cand2en as c2e

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def feature(words, kv, enc, mention_stat, id2title, llfile):
    sent = ' '.join(words)
    X_list = []
    cand_list = []
    gram_list = []
    mentions = {}
    for mention in words:
        candidates = c2e.mention2encands(
            mention, mention_stat, id2title, llfile)
        if candidates is None:
            continue
        encoded = eenc.encode_with_linkprob(candidates, kv)
        tmp_X2 = []
        tmp_X4 = []
        tmp_cand = []
        tmp_gram = []
        for target in encoded:
            tmp_cand.append(target["candname"])
            tmp_X2.append(target["entvec"])
            tmp_X4.append([target["linkprob"]])
            tmp_gram.append(mention)
            mentions[mention] = True
        if tmp_X2:
            X_list.append(
                [
                    None,
                    np.array(tmp_X2),
                    None,
                    np.array(tmp_X4)
                ]
            )
            cand_list.append(tmp_cand)
            gram_list.append(tmp_gram)
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
    return X_list, cand_list, gram_list


def predict(model, X_list, cand_list, gram_list, threshold=0.5):
    out = []
    for X, cands, grams in zip(X_list, cand_list, gram_list):
        y_preds = [x[0] for x in model.predict(X)]
        if np.max(y_preds) < threshold:
            continue
        index = np.argmax(y_preds)
        out.append((cands[index], grams[index]))
    return out


if __name__ == "__main__":
    from marisa_trie import BytesTrie
    id2title = BytesTrie()
    id2title.load("../data/id2title.marisa")
    mention_stat = BytesTrie()
    mention_stat.load("../data/mention_stat_ja.marisa")
    dbpath = "../data/enwiki_page.db"
    kv = eenc.load("../entity_vector/enwiki_20180420_100d.bin")
    enc = lenc.Encoder()
    tagger = MeCab.Tagger(
        "-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
    model = load_model("../models/model_wiki.h5")

    sent = None
    while sent != "-1":
        sent = input("sent>")
        words = tagger.parse(sent).split()
        X_list, cand_list, gram_list = feature(words, kv, enc, mention_stat,
                                               id2title, dbpath)
        result = predict(model, X_list, cand_list, gram_list)
        print(result)
