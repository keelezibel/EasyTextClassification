import os
import json
import pandas as pd
from bs4 import BeautifulSoup
import random
from tqdm import tqdm
import numpy as np
from nltk.tokenize import sent_tokenize

imperatives_csv_path = "/data/train/imperatives/imperatives.csv"
wikipedia_imperatives_path = "/data/train/imperatives/Wikipedia_AfD_imperative_data.txt"
spaadia_path = "/data/train/spaadia"
squadv2_path = "/data/train/squadv2/train-v2.0.json"

overall_dataset = dict()
category_count = dict()


def shuffle_data(data_split=0.8):
    comments = []
    categories = []

    comments_list = list(overall_dataset.items())
    random.shuffle(comments_list)

    for comment, category in comments_list:
        comments.append(comment.rstrip())
        categories.append(category)

    # Determine the training sample split point
    training_sample = int(len(comments) * data_split)

    # Split the dataset into training vs testing datasets
    x_train = np.array(comments[:training_sample])
    x_test = np.array(comments[training_sample:])
    y_train = np.array(categories[:training_sample])
    y_test = np.array(categories[training_sample:])

    return x_train, x_test, y_train, y_test


def proc_imperatives():
    imperative_count = 0
    # Pulls all the data from the manually generated imparatives dataset
    with open(imperatives_csv_path, "r") as imperative_file:
        for row in imperative_file:
            overall_dataset[str(row)] = "imperative"
            imperative_count += 1

    with open(wikipedia_imperatives_path, "r") as wiki_imperative_file:
        for row in wiki_imperative_file:
            label = row.strip()[-1]
            sentence = row.strip()[:-1]
            if int(label) == 1:
                overall_dataset[sentence] = "imperative"
                imperative_count += 1
    category_count["imp"] = imperative_count


def proc_spaadia():
    imperative_count = 0
    declarative_count = 0
    question_count = 0
    # Pulls all data from the SPAADIA dataset, adds to our dataset
    for doc in os.listdir(spaadia_path):
        with open(os.path.join(spaadia_path, doc), "r") as handle:
            conversations = BeautifulSoup(handle, features="xml")
            for imperative in conversations.findAll("imp"):
                overall_dataset[imperative.get_text().strip()] = "imperative"
                imperative_count += 1
            for declarative in conversations.findAll("decl"):
                overall_dataset[declarative.get_text().strip()] = "declarative"
                declarative_count += 1
            for question in conversations.findAll("q-yn"):
                overall_dataset[question.get_text().strip()] = "question"
                question_count += 1
            for question in conversations.findAll("q-wh"):
                overall_dataset[question.get_text().strip()] = "question"
                question_count += 1
    category_count["imp"] = category_count.get("imp", 0) + imperative_count
    category_count["dec"] = category_count.get("dec", 0) + declarative_count
    category_count["qn"] = category_count.get("qn", 0) + question_count


def proc_squadv2(n_samples=10000):
    qn_count = 0
    dec_count = 0
    with open(squadv2_path, "r") as f:
        json_data = json.load(f)
    df_context = pd.json_normalize(json_data["data"], record_path=["paragraphs"])
    for _, record in df_context.iterrows():
        context_entries = sent_tokenize(record["context"])
        for entry in context_entries:
            if not entry.strip():
                continue
            if dec_count >= n_samples:
                break
            overall_dataset[entry.strip()] = "declarative"
            dec_count += 1

    n_samples = df_context.shape[0]
    df = pd.json_normalize(json_data["data"], record_path=["paragraphs", "qas"])
    df_questions = df.sample(n=n_samples)
    for _, record in df_questions.iterrows():
        overall_dataset[record["question"]] = "question"
        qn_count += 1

    category_count["dec"] = category_count.get("dec", 0) + dec_count
    category_count["qn"] = category_count.get("qn", 0) + qn_count


def write_data(
    train_rec_path, test_rec_path, label_path, x_train, x_test, y_train, y_test
):
    # Convert to CSV format (text, label)
    train_rec = list()
    for text, label in zip(x_train, y_train):
        rec = {"text": text, "label": label}
        train_rec.append(rec)

    test_rec = list()
    for text, label in zip(x_test, y_test):
        rec = {"text": text, "label": label}
        test_rec.append(rec)

    df_train = pd.DataFrame(data=train_rec, columns=["text", "label"])
    df_test = pd.DataFrame(data=test_rec, columns=["text", "label"])

    df_train.to_csv(train_rec_path, index=False)
    df_test.to_csv(test_rec_path, index=False)

    # Export labels
    train_labels = df_train["label"].unique()
    test_labels = df_test["label"].unique()
    assert set(train_labels) == set(test_labels)

    labels = dict()
    for idx, label in enumerate(train_labels):
        labels[label] = idx
    with open(label_path, "w") as f:
        json.dump(labels, f, ensure_ascii=False)


def print_count():
    print(f"#Imperatives: {category_count['imp']}")
    print(f"#Declaratives: {category_count['dec']}")
    print(f"#Questions: {category_count['qn']}")


if __name__ == "__main__":
    proc_squadv2()
    proc_spaadia()
    proc_imperatives()
    print_count()

    x_train, x_test, y_train, y_test = shuffle_data()

    train_rec_path = "/data/train_rec.csv"
    test_rec_path = "/data/test_rec.csv"
    label_path = "/data/categories.json"
    write_data(
        train_rec_path, test_rec_path, label_path, x_train, x_test, y_train, y_test
    )
