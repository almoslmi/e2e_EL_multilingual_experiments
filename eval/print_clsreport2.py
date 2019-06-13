import json
from tqdm import tqdm

if __name__ == "__main__":
    with open("./result2.json") as f:
        data = json.load(f)
    recalls = []
    precisions = []

    for sent in tqdm(data):
        true_in_pred = 0
        pred_in_true = 0
        if len(sent["pred"]) == 0 or len(sent["true"]) == 0:
            continue
        sent["pred"] = [x[0] for x in sent["pred"]]
        for e in sent["true"]:
            if e in sent["pred"]:
                true_in_pred += 1
        precisions.append(float(true_in_pred)/float(len(sent["pred"])))
        for e in sent["pred"]:
            if e in sent["true"]:
                pred_in_true += 1
        recalls.append(float(pred_in_true)/float(len(sent["true"])))
    recall = sum(recalls)/len(recalls)
    precision = sum(precisions)/len(precisions)
    f1 = (2 * precision * recall) / (precision + recall)
           
    print("recall:", recall)
    print("precision:", precision)
    print("f1:", f1)
