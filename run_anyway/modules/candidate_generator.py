from marisa_trie import BytesTrie
import json


def load(datafile="../data/mention_stat.marisa"):
    trie = BytesTrie()
    trie.load(datafile)
    return trie


def generate(mention, trie, k=30):
    try:
        tmp_json = trie[mention]
        tmp_json = tmp_json[0].decode("utf-8")
        out = json.loads(tmp_json)
        out = sorted(out.items(), key=lambda x: x[1], reverse=True)[:k]
        return out
    except KeyError:
        return []


def generate_with_linkprob(mention, trie, k=30):
    try:
        tmp_json = trie[mention]
        tmp_json = tmp_json[0].decode("utf-8")
        out = json.loads(tmp_json)
        out = sorted(out.items(), key=lambda x: x[1], reverse=True)[:k]
        total = 0
        for x in out:
            total += x[1]
        out2 = []
        for i, x in enumerate(out):
            out2.append((x[0], float(x[1]) / float(total)))
        return out2
    except KeyError:
        return []


if __name__ == "__main__":
    import sys
    mention = sys.argv[1]
    trie = load()
    #print(generate(mention, trie))
    print(generate_with_linkprob(mention, trie))
