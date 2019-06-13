import json
import sys
from marisa_trie import BytesTrie

if __name__ == "__main__":
    print("load mention_stat")
    with open("./mention_stat.json") as f:
        data = json.load(f)

    print("mention_stat to trie")
    trie = BytesTrie([(k, bytes(json.dumps(v), "utf-8"))
                      for k, v in data.items()])

    print("saving...")
    trie.save("mention_stat.marisa")

    print("Done!")
