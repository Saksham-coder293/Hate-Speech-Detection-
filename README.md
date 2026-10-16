# Hate Speech Detection

Classifies text as **Hate Speech**, **Offensive Language**, or **Neither**, served as a REST API
and a small web interface.

**Live demo:** _<add your URL>_  |  **API docs:** _<your URL>_/docs

## Results (20% stratified hold-out, 4,957 tweets)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Hate Speech | 0.35 | 0.60 | 0.44 |
| Neither | 0.80 | 0.95 | 0.86 |
| Offensive Language | 0.97 | 0.88 | 0.92 |

Accuracy 87.2%, macro-F1 0.742. _(Update these from your own `artifacts/metrics.json`.)_

The first version (CountVectorizer + Decision Tree) reached ~87.5% accuracy but caught only ~34% of
hate speech, because ~77% of the data is "Offensive Language". Switching to TF-IDF (1-2 grams) +
class-balanced Logistic Regression, tuned on macro-F1, raised hate-speech recall to ~60%.

## Stack
scikit-learn, NLTK, FastAPI, Gradio, Docker, AWS EC2, Nginx.

## Run locally
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python train.py --data "labeled dataset.csv"   # creates artifacts/
python demo.py                                 # web UI only
uvicorn app:app --reload                       # UI at / and API at /predict, /docs
```

## API
```bash
curl -X POST https://<your-domain>/predict -H "Content-Type: application/json" \
  -d '{"text": "have a great day"}'
```

## Dataset
Davidson, Warmsley, Macy, Weber (2017), *Automated Hate Speech Detection and the Problem of
Offensive Language*, ICWSM. Download the labeled data from the authors' repository
(_<add link>_) and pass it to `train.py`. The dataset is not included here.

## Limitations
- Hate speech is the hardest class: precision ~0.35, so many flags are false positives.
- Labels are noisy, and the dataset is known to over-flag African-American English.
- Bag-of-words features miss context, sarcasm, and reclaimed language.
- Prototype only: not suitable for automated moderation. Inputs are not stored.
