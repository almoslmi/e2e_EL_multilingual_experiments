# coding: utf-8
import re
from tqdm import tqdm

def extract_mention_and_entity(exp):
    tmp = exp[2:-2]
    tmp2 = tmp[0].upper() + tmp[1:]
    if "|" in tmp2:
        entity, mention = tmp2.split("|")
        mention = mention.strip()
    else:
        entity = tmp2[:]
        mention = tmp[:]
    entity = entity.strip()
    entity = entity.replace(" ", "_")
    return entity, mention

if __name__ == "__main__":
    import json
    reg = re.compile(r"\[\[.+?\]\]")
    out = {}
    with open("dump", errors='ignore') as f:
        for line in tqdm(f):
            exps = re.findall(reg, line)
            for exp in exps:
                try:
                    entity, mention = extract_mention_and_entity(exp)
                except:
                    continue
                if mention in out:
                    if entity in out[mention]:
                        out[mention][entity] += 1
                    else:
                        out[mention][entity] = 1
                else:
                    out[mention] = {}

    with open("mention_stat.json", "w") as f:
        json.dump(out, f)
