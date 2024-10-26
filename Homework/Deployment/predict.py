import pickle



# gpt improved: have a function to load dv and model objects

def load_model():
    """
    loads dict vectorizer and model from pickle
    """
    with open("dv.bin", "rb") as f:
        dv = pickle.load(f)
    with open("model1.bin", "rb") as f:
        model = pickle.load(f)
    return dv, model

def predict_single(customer, dv, model):
    """
    predicts probability that client gets a subscription
    input: client as a dictionary
    output: probability (float)
    """
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1] #assuming get a subscription = 1
    return y_pred[0]

# print test prediction upon running
if __name__=="__main__":
    dv, model = load_model()
    pred = predict_single(customer={"job": "management", "duration": 400, "poutcome": "success"}
                        , dv=dv, model=model)
    print(f'Probability = {round(pred,2)}')
