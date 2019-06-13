import os
import sys
import json
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sys.path.append("./modules")

import candidate_generator as cg
import entvec_encoder as eenc
import laserencoder as lenc
import predictor as pr


def run():
    out = []
    with open("./data.json") as f:
        data = json.load(f)[946:1163]
        sents = []
        for doc in data:
            sents += doc
    model = load_model("./models/model_best.h5")
    trie = cg.load("./data/mention_stat.marisa")
    kv = eenc.load("./entity_vector/enwiki_20180420_100d.bin")
    enc = lenc.Encoder()

    for d in tqdm(sents):
        X_list, cand_list, gram_list = pr.feature(d["sentence"], trie, kv, enc)
        result = pr.predict(model, X_list, cand_list, gram_list)
        out.append({"true": d["entities"], "pred": result})

    with open("result.json", "w") as f:
        json.dump(out, f, indent=4)

        
if __name__ == "__main__":
    run()
