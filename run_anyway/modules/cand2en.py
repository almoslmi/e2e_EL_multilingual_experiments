import json
import sqlite3


def cand2en(cand, id2title, llfile):
    conn = sqlite3.connect(llfile)
    sql = """
select ll_from from langlinks where ll_title=?
"""
    c = conn.cursor()
    c.execute(sql, (cand, ))
    try:
        idx = c.fetchone()[0]
        return id2title[str(idx).encode()][0].decode('utf-8')
    except TypeError:
        return None
    except KeyError:
        return None


def mention2encands(mention,
                    wbegin,
                    wend,
                    mention_stat,
                    id2title,
                    llfile):
    try:
        cands = json.loads(mention_stat[mention][0])
        out = []
        total = 0
        for cand, count in cands.items():
            target = cand2en(cand, id2title, llfile)
            if target is not None and "(disambiguation)" not in target:
                out.append((target, count))
                total += count
        for i, (target, count) in enumerate(out):
            out[i] = (target, float(count)/float(total), (wbegin, wend))
        return out
    except KeyError:
        return None
    

if __name__ == "__main__":
    import sys
    from marisa_trie import BytesTrie
    mention = sys.argv[1]
    id2title = BytesTrie()
    id2title.load("../data/id2title.marisa")
    mention_stat = BytesTrie()
    mention_stat.load("../data/mention_stat_ja.marisa")
    dbpath = "../data/enwiki_page.db"
    print(mention2encands(mention, mention_stat, id2title, dbpath))
