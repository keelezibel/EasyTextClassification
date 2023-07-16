# Purpose of this repository

The purpose of this project is to provide boilerplate codes all the current state-of-the-art methods to classify text. Keep it simple approach, you can use the simplest method to LLM tuning method. Figure out which works for your setup and make it happen.

Libraries:
- Spacy textcat
- Spacy Setfit (WIP)
- RoBERTa (WIP)
- XLNet (WIP)
- LLM (p-tuning) (WIP)
- Wizmap (for visualization)

Environment:
- Docker
- > CUDA 12 on your host

## Spacy 

### Prepare your training dataset:

Use the following format:
- train.csv
- test.csv
- categories.json

`train.csv` and `test.csv` format
```
text,    label
<text>,  imperative
```

`categories.json` format (extend as many labels as you need)
```
{"label1": 0, "label2": 1, "label3": 2}
```

Put the three files into `src/textcat_spacy/assets`

### Modify `src/textcat_spacy/project.yml`
```
vars:
  name: "sentence_classifier"
  # Supported languages: all except ja, ko, th, vi, and zh, which would require
  # custom tokenizer settings in config.cfg
  lang: "en"       # Change your language accordingly, make sure it is supported by spacy
  # Set your GPU ID, -1 is CPU
  gpu_id: 0        # Change to whichever GPU you are using
  version: "1.0.0"
  train: "train_rec.csv"    # Change to your train CSV filename
  dev: "test_rec.csv"       # Change to your test CSV filename
  categories: "categories.json" # Change to your categories JSON filename
  config: "config.cfg"
```

### Train Model
Run the following commands to convert your dataset to spacy format, train and evaluate. Finally package into a whl for deployment
```python
python -m spacy project run convert
python -m spacy project run train
python -m spacy project run evaluate
python -m spacy project run package
pip install packages/<package-name>-1.0.0/dist/<package-name>-1.0.0.tar.gz
```

### Test your package
```python
python3 src/utils/sample_test_spacy_textcat.py
```

### Push your package
[WIP]

## Visualize your text using WizMap (credits to https://github.com/poloclub/wizmap)
```
git clone https://github.com/poloclub/wizmap.git`
npm run install
npm run dev
```

### Generate embeddings sample
`python src/wizmap_embedding/wizmap_viz.py`

### Run `data/wizmap/simple_server.py` to host your json and ndjson files