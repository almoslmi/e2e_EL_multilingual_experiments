from gensim.models import KeyedVectors
import candidate_generator as cg


def load(datafile="../entity_vector/enwiki_20180420_100d.bin"):
    kv = KeyedVectors.load(datafile, mmap="r")
    return kv


def encode(candidates, kv):
    out = []
    for candidate in candidates:
        try:
            out.append({
                "candname": candidate,
                "entvec": kv["ENTITY/" + candidate]
            })
        except KeyError:
            continue
    return out


def encode_with_linkprob(candidates, kv):
    out = []
    for candidate, prob in candidates:
        try:
            out.append({
                "candname": candidate,
                "entvec": kv["ENTITY/" + candidate],
                "linkprob": prob
            })
        except KeyError:
            continue
    return out


def encode_with_linkprob_and_windex(candidates, kv):
    out = []
    for candidate, prob, wind in candidates:
        try:
            out.append({
                "candname": candidate,
                "entvec": kv["ENTITY/" + candidate],
                "linkprob": prob,
                "begin": wind[0],
                "end": wind[1]
            })
        except KeyError:
            continue
    return out


if __name__ == "__main__":
    import sys
    mention = sys.argv[1]
    trie = cg.load()
    kv = load()
    candidates = [k for k, _ in cg.generate(mention, trie)]
    print(encode(candidates, kv))
