import re, json
from pathlib import Path

LABELS = [
    "O","PATIENT","AGE","DATE","PHONE","EMAIL","ID","LOCATION","HOSPITAL","DOCTOR"
]
LABEL2ID = {x:i for i,x in enumerate(LABELS)}
ID2LABEL = {i:x for x,i in LABEL2ID.items()}

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def tokenize_with_offsets(text):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

def bio_from_entities(text, entities):
    toks = tokenize_with_offsets(text)
    labels = ["O"] * len(toks)
    for ent in entities:
        s,e,lab = ent["start"], ent["end"], ent["label"]
        hit = [i for i,(_,ts,te) in enumerate(toks) if te > s and ts < e]
        for j,i in enumerate(hit):
            labels[i] = ("B-" if j==0 else "I-") + lab
    return toks, labels

def load_jsonl(path):
    data=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def strip_bio(label):
    return label.split("-",1)[-1] if "-" in label else label
