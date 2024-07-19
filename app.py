from flask import Flask, request, jsonify, render_template
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Path to the local tokenizer directory
LOCAL_TOKENIZER_PATH = r"C:\Users\yashk\OneDrive\Desktop\Project\smart-recipe"

# Path to the local model directory
LOCAL_MODEL_PATH = r"C:\Users\yashk\OneDrive\Desktop\Project\smart-recipe"

# Load tokenizer from the local directory
tokenizer = AutoTokenizer.from_pretrained(LOCAL_TOKENIZER_PATH, use_fast=True)

# Load model from the local directory
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH)

prefix = "items: "
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")

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

def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs, 
        max_length=256, 
        padding="max_length", 
        truncation=True, 
        return_tensors="jax"
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(generated, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    data = request.form
    items = [data.get('ingredients', '')]
    generated = generation_function(items)
    recipe_sections = []
    for text in generated:
        sections = text.split("\n")
        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                section = section.replace("title:", "")
                headline = "TITLE"
            elif section.startswith("ingredients:"):
                section = section.replace("ingredients:", "")
                headline = "INGREDIENTS"
            elif section.startswith("directions:"):
                section = section.replace("directions:", "")
                headline = "DIRECTIONS"

            if headline == "TITLE":
                recipe_sections.append(f"[{headline}]: {section.strip().capitalize()}")
            else:
                section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
                recipe_sections.append(f"[{headline}]:")
                recipe_sections.extend(section_info)

    return render_template('index.html', ingredients=data.get('ingredients', ''), recipe_sections=recipe_sections)

if __name__ == '__main__':
    app.run(debug=True)
