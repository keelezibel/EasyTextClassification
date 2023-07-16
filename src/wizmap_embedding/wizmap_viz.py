import os
from glob import glob
from os.path import exists, join, basename
from tqdm import tqdm
from json import load, dump
from matplotlib import pyplot as plt
from collections import Counter
from umap import UMAP
import wizmap
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize


class WizMap:
    def __init__(self, model_name="/models/sentence_transformers/all-mpnet-base-v2"):
        self.batch_size = 128
        self.model = SentenceTransformer()

    def embed_data(self, data_folder):
        files = glob(os.path.join(data_folder, f"*.txt"))
        n_sentences = 3

        all_text = []
        for file in files:
            with open(file, "r") as f:
                content = f.read()
                sentences = sent_tokenize(content)
                merged_sentences = [
                    "".join(map(str, sentences[i : i + n_sentences]))
                    for i in range(0, len(sentences), n_sentences)
                ]
                all_text.extend(merged_sentences)

        embeddings = self.model.encode(
            all_text, batch_size=self.batch_size, show_progress_bar=True
        )

        print(f"Loaded {len(all_text)} text")
        print(f"Embedding shape: {embeddings.shape}")

        return all_text, embeddings

    def dim_reduction(self, embeddings):
        # UMAP Dimension Reduction
        reducer = UMAP(metric="cosine")
        embeddings_2d = reducer.fit_transform(embeddings)

        xs = embeddings_2d[:, 0].astype(float).tolist()
        ys = embeddings_2d[:, 1].astype(float).tolist()

        return xs, ys

    def generate_json(self, texts, xs, ys):
        data_list = wizmap.generate_data_list(xs, ys, texts)
        grid_dict = wizmap.generate_grid_dict(xs, ys, texts, "IMDB Reviews")

        # Save the JSON files
        wizmap.save_json_files(data_list, grid_dict, output_dir="/data")


if __name__ == "__main__":
    wizmap_obj = WizMap()

    subfolder = "<subfolder-name>"
    data_folder = f"/data/test/{subfolder}"
    all_text, embeddings = wizmap_obj.embed_data(data_folder)
    xs, ys = wizmap_obj.dim_reduction(embeddings)
    wizmap_obj.generate_json(all_text, xs, ys)
