# coding: utf-8
from gensim.models import KeyedVectors
kv = KeyedVectors.load("./enwiki_20180420_100d.bin", mmap="r")
kv["Obama"]
kv["obama"]
kv.index2entity
kv.index2entity[0]
kv["obama_(city)"]
kv["obama,fukui"]
kv["obama,_fukui"]
data = {x:None for x in kv.index2entity}
data["obama"]
for x, y in data.items():
    if "obama" in x:
        print(x)
        
for x, y in data.items():
    if "Obama" in x:
        print(x)
        
kv["ENTITY/Python"]
kv["ENTITY/Obama_Day"]
kv["ENTITY/Python_(programming_language)"]
