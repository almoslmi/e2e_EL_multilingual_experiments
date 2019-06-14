# coding: utf-8
import re
from tqdm import tqdm

if __name__ == "__main__":
    import json
    reg = re.compile(r"\[\[(.+?)\|(.+?)\]\]")
    reg2 = re.compile(r"\{\{.*?\}\}")
    reg3 = re.compile(r"  +")
    reg4 = re.compile(r"\[\[(((?!\|).)*?)\]\]")
    out = {}
    with open("enwiki.xml", errors='ignore') as f:
        for i, line in tqdm(enumerate(f)):
            if line.replace(" ", "").startswith("<"):
                continue
            exps = re.findall(reg, line)
            exps2 = re.findall(reg4, line)
            ents = []
            for exp in exps:
                entity = exp[0]
                if len(entity) < 2:
                    continue
                entity = entity[0].upper() + entity[1:]
                entity = entity.strip().replace(" ", "_")
                ents.append(entity)
            for exp in exps2:
                entity = exp[0]
                if len(entity) < 2:
                    continue
                entity = entity[0].upper() + entity[1:]
                entity = entity.strip().replace(" ", "_")
                ents.append(entity)

            line = re.sub(reg2, " ", line)
            line = re.sub(reg, r"\2", line)
            line = re.sub(reg4, r"\1", line)
            line = line.replace("\n", " ")
            line = re.sub(reg3, " ", line)
            line = line + "\t" + ':::'.join(ents)

            in_wrongsym = False
            in_wrongsym = in_wrongsym or "|" in line
            in_wrongsym = in_wrongsym or "[" in line or "]" in line
            in_wrongsym = in_wrongsym or "{" in line or "}" in line
            in_wrongsym = in_wrongsym or "=" in line or "'" in line
            in_wrongsym = in_wrongsym or "\n" in line
            in_wrongsym = in_wrongsym or ":::" not in line
            in_wrongsym = in_wrongsym or "&" in line
            in_wrongsym = in_wrongsym or "Category" in line
            in_wrongsym = in_wrongsym or "*" in line

            if in_wrongsym:
                continue
            with open("ELdata_wiki.txt", "a") as f:
                f.write(line + "\n")
