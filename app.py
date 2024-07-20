from flask import Flask, request, render_template
import numpy as np
import joblib

churn_model = joblib.load('model_svc.pkl')
scaler = joblib.load('scalerf.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_churn():
   
    credit_score = int(request.form['credit_score'])
    country_st = request.form['country']
    if(country_st=="France"):
        country=0
    elif(country_st=="Spain"):
        country=2
    else:
        country=1
    
   
    gender_st = request.form['gender']
    if(gender_st=="female"):
        gender=0
    else:
        gender=1
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    balance = int(request.form['balance'])
    products_number = int(request.form['products_number'])
    
    credit_card_st = request.form['credit_card']
    if(credit_card_st=="yes"):
        credit_card=1
    else:
        credit_card=0
    active_member_st = request.form['active_member']
    
    if(active_member_st=="yes"):
        active_member=1
    else:
        active_member=0
    estimated_salary = int(request.form['estimated_salary'])

   
    cols_sc = [[credit_score,age,tenure,balance,products_number,estimated_salary]]
# print(cols_sc)
    scaled_cols = scaler.transform(cols_sc)
# print(scaled_cols[0])
    feature_vector = [[scaled_cols[0][0], country, gender, scaled_cols[0][1], scaled_cols[0][2], scaled_cols[0][3], scaled_cols[0][4], credit_card, active_member, scaled_cols[0][5]]]
# print(feature_vector)


# Make predictions using the loaded model
    prediction = churn_model.predict(feature_vector) 
    return render_template("result.html",result=prediction)


if __name__ == '__main__':
    app.run(debug=True)
