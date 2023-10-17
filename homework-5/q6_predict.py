import pickle

from flask import Flask
from flask import request
from flask import jsonify


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)

dv = load('dv.bin')
model = load('model2.bin')

app = Flask('credit')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]
    grant_credit = y_pred >= 0.5

    result = {
        'credit_probability': float(y_pred),
        'grant_credit': bool(grant_credit)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)