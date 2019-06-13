import json

def run(datafile="./data.json"):
    with open(datafile) as f:
        data = json.load(f)

    sent_counts = []
    current = 0
    for i, doc in enumerate(data):
        current += len(doc)
        if i == 945 or i==1161 or i==1392:
            sent_counts.append(current)
    print(sent_counts)

if __name__ == "__main__":
    run()
