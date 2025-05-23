from flask import Flask, request, jsonify
from functions import prompt_analysis, save_prompt_history, data

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    prompt_analysis(prompt)
    result = data.iloc[-1].to_dict()
    return jsonify(result)

@app.route('/save', methods=['POST'])
def save():
    save_prompt_history()
    return jsonify({"message": "Prompt history saved."})

if __name__ == "__main__":
    app.run()
