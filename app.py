from flask import Flask,json,jsonify,request,render_template
from predict import make_predictions, preprocess_input

app= Flask(__name__) #special variable

# user_input={
#     "Age":1,
#     "Sex":"male",
#     "Job":2
# }

# user_input={'Age': 23,
#   'Sex': 'female',
#   'Job': 1,
#   'Housing': 'rent',
#   'Saving accounts': None,
#   'Checking account': None,
#   'Credit amount': 3234,
#   'Duration': 24,
#   'Purpose': 'furniture/equipment',
#   'Risk': 'bad'
#   }

@app.route('/',methods=['GET']) #
def hello():
    return "Hello World"

@app.route('/predict',methods=["POST"]) #
def predict():
    if request.method=='POST':
        data=request.get_json()
        # result=preprocess_input(data)
        result=make_predictions(data)

        
        result={
        "model":"LR",
        "api_version":"v1",
        "result":str(round(list(result)[0],3))
        }

        # result=result.to_dict('records')
        
    return jsonify(result)

if __name__== "__main__":
    app.run(port=5000, debug=False)