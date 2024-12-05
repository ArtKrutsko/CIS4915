from flask import Flask, request, jsonify
from flask_cors import CORS
from request_format import process_request

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    # Example response
    data = request.json
    print("Processing request: ", data)
    
    result = process_request(data)
    
    # sample output
    # result = {
    #     "output": [
    #         {"type": "text", "content": f"Data input: {data}"},
    #         {"type": "image", "content": f"output.png"}
    #     ]
    # }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
