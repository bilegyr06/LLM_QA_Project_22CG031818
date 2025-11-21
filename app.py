from flask import Flask, render_template, request, jsonify
from LLM_QA_CLI import preprocess_text, query_llm
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Process the question
    processed_question = preprocess_text(question)
    
    # Get answer from LLM
    # Note: We are sending the processed question to the LLM as per the CLI logic we established.
    # If you prefer sending the raw question for better quality, you can change this to query_llm(question)
    answer = query_llm(processed_question)
    
    return jsonify({
        'processed_question': processed_question,
        'answer': answer
    })

if __name__ == '__main__':
    # Run the app
    # debug=True for development
    app.run(debug=True, port=5000)
