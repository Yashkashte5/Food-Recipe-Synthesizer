import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from datasets import load_metric

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
import nltk.translate.bleu_score as bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk.translate.gleu_score as gleu
import nltk.translate.meteor_score as meteor

from jiwer import wer, mer

import re
import math
from collections import Counter
import string
from tqdm import tqdm


nltk.download('stopwords')
stopwords = stopwords.words("english")


df = pd.read_csv("./test_generated.csv", sep="\t")
true_recipes = df["true_recipe"].values.tolist()
generated_recipes = df["generated_recipe"].values.tolist()

def cleaning(text, rm_sep=True, rm_nl=True, rm_punk_stopwords=True):
    if rm_sep:
        text = text.replace("--", " ")
    
    if rm_nl:
        text = text.replace("\n", " ")

    if rm_punk_stopwords:
        text = " ".join([word.strip() for word in wordpunct_tokenize(text) if word not in string.punctuation and word not in stopwords and word])
    else:
        text = " ".join([word.strip() for word in wordpunct_tokenize(text) if word.strip()])

    text = text.lower()
    return text

X, Y = [], []
for x, y in tqdm(zip(true_recipes, generated_recipes), total=len(df)):
    x, y = cleaning(x, True, True, True), cleaning(y, True, True, True)

    if len(x) > 16 and len(y) > 16:
        X.append(x)
        Y.append(y)


print(f"Sample X: {X[0]}")
print(f"Sample Y: {Y[0]}")

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)

def get_result(content_a, content_b):
    text1 = content_a
    text2 = content_b

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result


cosim_scores = []
for i in tqdm(range(len(X))):
    cosim_scores.append(get_result(X[i], Y[i]))

cosim_score = np.array(cosim_scores).mean()
print(f"Cosine similarity score: {cosim_score}")  # 0.714542

X, Y = [], []
for x, y in tqdm(zip(true_recipes, generated_recipes), total=len(df)):
    x, y = cleaning(x, True, True, False), cleaning(y, True, True, False)

    if len(x) > 16 and len(y) > 16:
        X.append(x)
        Y.append(y)
        
        
wer = load_metric("wer")
wer_score = wer.compute(predictions=Y, references=X)
print(f"WER score: {wer_score}")  # 0.70938

    
rouge = load_metric("rouge")
rouge_score = rouge.compute(predictions=Y, references=X, use_stemmer=True)
rouge_score = {key: value.mid.fmeasure * 100 for key, value in rouge_score.items()}
print(f"Rouge score: {rouge_score}")  # {'rouge1': 56.30779082900833, 'rouge2': 29.07704230163075, 'rougeL': 45.812165960365924, 'rougeLsum': 45.813971137090654}

bleu = load_metric("bleu")
def postprocess_text(preds, labels):
    preds = [wordpunct_tokenize(pred) for pred in preds]
    labels = [[wordpunct_tokenize(label)] for label in labels]

    return preds, labels

Y, X = postprocess_text(Y, X)
bleu_score = bleu.compute(predictions=Y, references=X)["bleu"]
print(f"BLEU score: {bleu_score}")  # 0.203867