from flask import Flask, render_template, request, jsonify
import time
import myllm, Config

app = Flask(__name__)

class QueryEngine:
    def query(self, input_str):
        # Dummy method, replace this with your actual query logic
        return {"response": "Sample Response", "metadata": {"sql_query": "SELECT * FROM dummy", "result": "Sample Result"}}

query_engine = QueryEngine()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    try:
        response = query_engine.query(user_input)

        print("\nMayaAI: ", response)
        print(" \nMetadata Info:")
        print("     MayaSQL:", response.metadata['sql_query'])
        print("     MayaSQLResult:", response.metadata['result'])

        return jsonify({"response": response, "metadata": response.metadata})
    except Exception as e:
        return jsonify({"error": f"ERROR --- SQL {e}. Please modify the question so that it's only related to one table."})

if __name__ == '__main__':
    app.run(debug=True)
