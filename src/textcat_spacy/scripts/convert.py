"""Convert textcat annotation from JSONL to spaCy v3 .spacy format."""
import csv
import json
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def convert(lang: str, input_path: Path, cats_json: Path, output_path: Path):
    with open(cats_json, "r") as f:
        cats = json.load(f)
    one_hot_dicts = {}
    for c in cats:
        one_hot_dict = {t: (1 if t == c else 0) for t in cats}
        one_hot_dicts[c] = one_hot_dict

    nlp = spacy.blank(lang)
    db = DocBin()
    with open(input_path, "r") as f:
        reader = csv.reader(f)
        hdr = next(reader)
        for row in reader:
            doc = nlp.make_doc(row[0])
            doc.cats = one_hot_dicts[row[1]]
            db.add(doc)
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
