import os
import spacy
from glob import glob
from tqdm import tqdm
from nltk.tokenize import sent_tokenize


class SampleRecordClassifier:
    def __init__(self, model_name="en_sentence_classifier") -> None:
        self.nlp = spacy.load(model_name)  # Change to your package name

    def infer(self, test_rec, test_cat):
        count = 0
        assert len(test_rec) == len(test_cat)
        for text, label in zip(test_rec, test_cat):
            pred = self.nlp(text)
            pred_label = max(pred.cats, key=pred.cats.get)
            if pred_label == label:
                count += 1
            print(f"{text}, {label}, {pred_label}")
        print(f"Accuracy: {count/len(test_rec)*100.0}")


class SampleFilesClassifier:
    def __init__(self, model_name="en_sentence_classifier") -> None:
        self.nlp = spacy.load(model_name)  # Change to your package name

    def infer(self, data_folder, sub_folder, label):
        files = glob(os.path.join(data_folder, f"*.txt"))
        label_rec = []
        n_sentences = 3
        for file in tqdm(files):
            with open(file, "r") as f:
                content = f.read()
                sentences = sent_tokenize(content)
                merged_sentences = [
                    "".join(map(str, sentences[i : i + n_sentences]))
                    for i in range(0, len(sentences), n_sentences)
                ]

                for sentence in merged_sentences:
                    pred = self.nlp(sentence)
                    pred_label = max(pred.cats, key=pred.cats.get)
                    if pred_label == label:
                        label_rec.append(sentence)

        print(f"Found {len(label_rec)} label_rec sentences")
        with open(os.path.join("/data/test", f"{sub_folder}_imp.txt"), "w") as f:
            for line in label_rec:
                f.write(f"{line}\n")


if __name__ == "__main__":
    test_rec = [
        "What was Prince Albert 's nickname ?",
        "The centre of London is said to be located by the Eleanor Cross in Charing Cross near the junction of Trafalgar Square and Whitehall .",
        "Don't ever touch my phone.",
    ]

    test_cat = ["question", "declarative", "imperative"]

    record_classifier = SampleRecordClassifier()
    record_classifier.infer(test_rec, test_cat)

    # # Uncomment for files classification only
    # sub_folder = "ZN"
    # data_folder = f"/data/test/{sub_folder}"
    # label = "imperative"
    # files_classifier = SampleFilesClassifier()
    # files_classifier.infer(data_folder, sub_folder, label)
