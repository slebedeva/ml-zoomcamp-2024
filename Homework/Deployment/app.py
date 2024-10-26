from flask import Flask, request, jsonify
from predict import load_model, predict_single 

# initiate Flask app
app = Flask("predict")

# load model and dv #gpt
dv, model = load_model()

# predict
# code from lecture
@app.route("/predict", methods=['POST'])
def predict():
    customer = request.get_json()

    #predict probability that this client will get a subscription
    prediction = predict_single(customer, dv, model)
    sub = prediction >= 0.5
    
    result = {
        'subscription_probability': float(prediction),
        'subscription': bool(sub),
    }
    # convert py dict to json
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
