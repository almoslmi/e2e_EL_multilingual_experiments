import simplejson as json
import numpy as np
import pickle
from tqdm import tqdm


def generate(indices, sentvecs, labels, entvecs):
    while True:
        for index in indices:
            X1 = []
            X2 = []
            y = []
            labs = labels[index]
            x1 = sentvecs[index]
            for lab in labs:
                X1.append(x1)
                X2.append(entvecs[lab[0]])
                y.append(lab[1])
            if y:
                yield [np.array(X1), np.array(X2)], np.array(y)


def generate_test(indices, sentvecs, labels, entvecs):
    X1 = []
    X2 = []
    y = []
    for index in tqdm(indices):
        labs = labels[index]
        x1 = sentvecs[index]
        for lab in labs:
            X1.append(x1)
            X2.append(entvecs[lab[0]])
            y.append(lab[1])
    return [np.array(X1), np.array(X2)], np.array(y)


def generate2(indices, sentvecs, labels, entvecs, gramvecs):
    while True:
        for index in indices:
            X1 = []
            X2 = []
            X3 = []
            X4 = []
            y = []
            labs = labels[index]
            x1 = sentvecs[index]
            for lab in labs:
                x2 = entvecs[lab[1]]
                x3 = gramvecs[lab[0]]
                x4 = [float(lab[2])]
                X1.append(x1)
                X2.append(x2)
                X3.append(x3)
                X4.append(x4)
                y.append(lab[3])
            if y:
                yield [np.array(X1),
                       np.array(X2),
                       np.array(X3),
                       np.array(X4)], np.array(y)


def generate2_test(indices, sentvecs, labels, entvecs, gramvecs):
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    y = []

    for index in indices:
        labs = labels[index]
        x1 = sentvecs[index]
        for lab in labs:
            x2 = entvecs[lab[1]]
            x3 = gramvecs[lab[0]]
            x4 = [float(lab[2])]
            X1.append(x1)
            X2.append(x2)
            X3.append(x3)
            X4.append(x4)
            y.append(lab[3])
    return [np.array(X1),
            np.array(X2),
            np.array(X3),
            np.array(X4)], np.array(y)

                
def generate3(indices, sentvecs, labels, entvecs, linkprobs):
    while True:
        for index in indices:
            X1 = []
            X2 = []
            y = []
            labs = labels[index]
            x1 = sentvecs[index]
            for lab in labs:
                X1.append(x1)
                X2.append(entvecs[lab[0]])
                y.append(lab[1])
            if y:
                yield [np.array(X1), np.array(X2)], np.array(y)


def load_data(svecfile="./training_data/sents_encoded.npy",
              labfile="./training_data/labels.json",
              evecfile="./training_data/ent_encoded.pkl"):
    sentvecs = np.load(svecfile)
    with open(labfile) as f:
        labels = json.load(f)
    with open(evecfile, "rb") as f:
        entvecs = pickle.load(f)
    return sentvecs, labels, entvecs


def load_data2(svecfile="./training_data/sents_encoded.npy",
               labfile="./training_data/labels2.json",
               evecfile="./training_data/ent_encoded.pkl",
               g2vecfile="./training_data/gram_encoded.pkl"):
    sentvecs = np.load(svecfile)
    with open(labfile) as f:
        labels = json.load(f)
    with open(evecfile, "rb") as f:
        entvecs = pickle.load(f)
    with open(g2vecfile, "rb") as f:
        gramvecs = pickle.load(f)
    return sentvecs, labels, entvecs, gramvecs


def gen_indices(num_sents):
    all_inds = list(range(num_sents))
    train_inds = all_inds[:12938]
    val_inds = all_inds[12938:13938]
    test_inds = all_inds[13938:17163]
    return train_inds, val_inds, test_inds
