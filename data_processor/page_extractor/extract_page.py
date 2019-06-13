import re
import pickle
from tqdm import tqdm
import mmap


def dumpiter(dumpfile):
    with open(dumpfile) as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        the_iter = re.finditer(b"\(([0-9]+),([0-9]+),'(.+?)'", m)
    return the_iter


def transform(the_iter):
    out = {}
    for d in tqdm(the_iter):
        if int(d.group(2)) == 0:
            out[d.group(1)] = d.group(3)
    return out


if __name__ == '__main__':
    out = transform(dumpiter("./enwiki-latest-page.sql"))
    with open("id2title.pkl", "wb") as f:
        pickle.dump(out, f)
