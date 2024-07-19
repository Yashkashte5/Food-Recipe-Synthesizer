import logging
import os
import pandas as pd
import random
import re
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key

from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm
import pandas as pd


print(jax.devices())

MODEL_NAME_OR_PATH = "../"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

prefix = "items: "
text_column = "inputs"
target_column = "targets"
max_source_length = 256
max_target_length = 1024
seed = 42
eval_batch_size = 64
# generation_kwargs = {
#     "max_length": 1024,
#     "min_length": 128,
#     "no_repeat_ngram_size": 3,
#     "do_sample": True,
#     "top_k": 60,
#     "top_p": 0.95
# }
generation_kwargs = {
    "max_length": 1024,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "num_beams": 4,
    "length_penalty": 1.5,
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, '')

    return text

def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]
    
    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)

        for k, v in tokens_map.items():
            text = text.replace(k, v)

        new_texts.append(text)

    return new_texts


predict_dataset = load_dataset("csv", data_files={"test": "/home/m3hrdadfi/code/data/test.csv"}, delimiter="\t")["test"]
print(predict_dataset)
# predict_dataset = predict_dataset.select(range(10))
# print(predict_dataset)
column_names = predict_dataset.column_names
print(column_names)


# Setting padding="max_length" as we need fixed length inputs for jitted functions
def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[target_column]
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs, 
        max_length=max_source_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="np"
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=max_target_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="np"
        )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

predict_dataset = predict_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=None,
    remove_columns=column_names,
    desc="Running tokenizer on prediction dataset",
)

def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch

rng = jax.random.PRNGKey(seed)
rng, dropout_rng = jax.random.split(rng)
rng, input_rng = jax.random.split(rng)

def generate_step(batch):
    output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **generation_kwargs)
    return output_ids.sequences

p_generate_step = jax.pmap(generate_step, "batch")

pred_generations = []
pred_labels = []
pred_inputs = []
pred_loader = data_loader(input_rng, predict_dataset, eval_batch_size)
pred_steps = len(predict_dataset) // eval_batch_size

for _ in tqdm(range(pred_steps), desc="Predicting...", position=2, leave=False):
    # Model forward
    batch = next(pred_loader)
    inputs = batch["input_ids"]
    labels = batch["labels"]

    generated_ids = p_generate_step(batch)
    pred_generations.extend(jax.device_get(generated_ids.reshape(-1, generation_kwargs["max_length"])))
    pred_labels.extend(jax.device_get(labels.reshape(-1, labels.shape[-1])))
    pred_inputs.extend(jax.device_get(inputs.reshape(-1, inputs.shape[-1])))

inputs = tokenizer.batch_decode(pred_inputs, skip_special_tokens=True)
true_recipe = target_postprocessing(
    tokenizer.batch_decode(pred_labels, skip_special_tokens=False), 
    special_tokens
)
generated_recipe = target_postprocessing(
    tokenizer.batch_decode(pred_generations, skip_special_tokens=False),
    special_tokens
)
test_output = {
    "inputs": inputs,
    "true_recipe": true_recipe,
    "generated_recipe": generated_recipe
}
test_output = pd.DataFrame.from_dict(test_output)
test_output.to_csv("./generated_recipes_b.csv", sep="\t", index=False, encoding="utf-8")
