# coding: utf-8
import pickle
from collections import defaultdict

from marisa_trie import BytesTrie
from tqdm import tqdm


if __name__ == "__main__":
    with open("./id2title.pkl", "rb") as f:
        data = pickle.load(f)

    trie_id2title = BytesTrie([(x.decode('utf-8'), y)
                               for x, y in tqdm(data.items())])
    trie_id2title.save("id2title.marisa")

    out = defaultdict(list)
    for k, v in tqdm(data.items()):
        out[v.decode('utf-8')].append(k.decode('utf-8'))

    del (data)
    del (trie_id2title)
    trie_title2id = BytesTrie([(x, y[0].encode()) for x, y in tqdm(out.items())
                               if len(y) == 1])
    trie_title2id.save("title2id.marisa")
