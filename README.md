# Food Recipe Synthesizer

A Python-based application that generates customized recipes based on available ingredients using a fine-tuned Hugging Face Seq2Seq model. The system allows users to input ingredients and get recipe suggestions in real-time.

## Features
- **Ingredient-based Recipe Generation**: Generates recipes tailored to the user's available ingredients.
- **Seq2Seq Model**: Fine-tuned Hugging Face model for accurate and contextually relevant recipe outputs.
- **Flask Web Application**: Provides a simple interface for users to input ingredients and receive recipes.
- **Tested Locally**: Fully functional locally for experimentation and model testing.
- **Notebook Support**: Includes `recipe.ipynb` for testing and experimentation with the model.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Yashkashte5/Food-Recipe-Synthesizer.git
   cd Food-Recipe-Synthesizer
   ```
2. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Enter your available ingredients and get recipe suggestions.

## Project Structure
```
Food-Recipe-Synthesizer/
│── app.py
│── config.json
│── recipe.ipynb
│── src/
│   ├── model/                # Model files and tokenizer
│   └── utils/                # Helper functions
└── templates/                # HTML templates for Flask app 
```

## Future Improvements
- Deploy as a web application for public use.
- Add more robust ingredient-to-recipe mapping for complex ingredient combinations.
- Incorporate user feedback for improving recipe accuracy.
- Expand dataset for more diverse cuisine coverage.
