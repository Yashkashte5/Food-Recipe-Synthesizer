import torch
import numpy as np
import jax.numpy as jnp

from transformers import AutoTokenizer
from transformers import FlaxT5ForConditionalGeneration
from transformers import T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("../")
model_fx = FlaxT5ForConditionalGeneration.from_pretrained("../")
model_pt = T5ForConditionalGeneration.from_pretrained("../", from_flax=True)
model_pt.save_pretrained("./")

text = "Hello To You"
e_input_ids_fx = tokenizer(text, return_tensors="np", padding=True, max_length=128, truncation=True)
d_input_ids_fx = jnp.ones((e_input_ids_fx.input_ids.shape[0], 1), dtype="i4") * model_fx.config.decoder_start_token_id

e_input_ids_pt = tokenizer(text, return_tensors="pt", padding=True, max_length=128, truncation=True)
d_input_ids_pt = np.ones((e_input_ids_pt.input_ids.shape[0], 1), dtype="i4") * model_pt.config.decoder_start_token_id


print(e_input_ids_fx)
print(d_input_ids_fx)

print()

encoder_pt = model_fx.encode(**e_input_ids_pt)
decoder_pt = model_fx.decode(d_input_ids_pt, encoder_pt)
logits_pt = decoder_pt.logits
print(logits_pt)

encoder_fx = model_fx.encode(**e_input_ids_fx)
decoder_fx = model_fx.decode(d_input_ids_fx, encoder_fx)
logits_fx = decoder_fx.logits
print(logits_fx)