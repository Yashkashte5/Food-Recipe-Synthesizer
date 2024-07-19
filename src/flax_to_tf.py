import torch
import numpy as np
import jax.numpy as jnp

from transformers import AutoTokenizer
from transformers import FlaxT5ForConditionalGeneration
from transformers import TFT5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("../")
model_fx = FlaxT5ForConditionalGeneration.from_pretrained("../")
model_tf = TFT5ForConditionalGeneration.from_pretrained("./", from_pt=True)
model_tf.save_pretrained("./")

text = "Hello To You"
e_input_ids_fx = tokenizer(text, return_tensors="np", padding=True, max_length=128, truncation=True)
d_input_ids_fx = jnp.ones((e_input_ids_fx.input_ids.shape[0], 1), dtype="i4") * model_fx.config.decoder_start_token_id

e_input_ids_tf = tokenizer(text, return_tensors="tf", padding=True, max_length=128, truncation=True)
d_input_ids_tf = np.ones((e_input_ids_tf.input_ids.shape[0], 1), dtype="i4") * model_tf.config.decoder_start_token_id


print(e_input_ids_fx)
print(d_input_ids_fx)

print()

encoder_tf = model_fx.encode(**e_input_ids_tf)
decoder_tf = model_fx.decode(d_input_ids_tf, encoder_tf)
logits_tf = decoder_tf.logits
print(logits_tf)

encoder_fx = model_fx.encode(**e_input_ids_fx)
decoder_fx = model_fx.decode(d_input_ids_fx, encoder_fx)
logits_fx = decoder_fx.logits
print(logits_fx)