# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import fitz  # PyMuPDF
from transformers import pipeline

app = Flask(__name__)
CORS(app)

qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

@app.route('/upload', methods=['POST'])
def upload_pdf():
    file = request.files['pdf']
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return jsonify({"text": text})

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    context = data['context']
    result = qa_pipeline(question=question, context=context)
    return jsonify({"answer": result['answer']})

if __name__ == '__main__':
    app.run(debug=True)