import sys
from keras.models import load_model
from sklearn.metrics import classification_report

sys.path.append("../train/")
from data_generator import load_data2, generate2_test, gen_indices


def run():
    pdata = load_data2(svecfile="../train/training_data/sents_encoded.npy",
                       labfile="../train/training_data/labels2.json",
                       evecfile="../train/training_data/ent_encoded.pkl",
                       g2vecfile="../train/training_data/gram_encoded.pkl")

    _, _, test_inds = gen_indices(len(pdata[1]))
    model = load_model("../train/model_aida_wiki_best.h5")
    X_test, y_test = generate2_test(test_inds, *pdata)
    y_preds = model.predict(X_test)
    y_preds = [y > 0.5 for y in y_preds]
    print(classification_report(y_test, y_preds))


if __name__ == "__main__":
    run()
