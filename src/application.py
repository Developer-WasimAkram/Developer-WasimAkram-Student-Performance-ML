from flask import Flask,request,render_template
from prediction_pipeline import PredictionPipeline,CustomData 
import numpy as np 
import pandas as pd   

application = Flask(__name__)
app=application
# Route for a homepage 


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods=['POST','GET'])
def predict_datapoint():
    # Extract the data from the form
    if request.method == 'GET':
        return render_template('predict.html')
    
    else:
        data=CustomData(
            
            gender = request.form['gender'],
            race_ethnicity = request.form['ethnicity'],
            parental_level_of_education = request.form['parental_level_of_education'],
            lunch = request.form['lunch'],
            test_preparation_course = request.form['test_preparation_course'],
            reading_score = float(request.form['reading_score']),
            writing_score = float(request.form['writing_score']),
            )

       # Create a prediction pipeline object
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(data.get_data_as_data_frame())
        print(prediction)  # Print the prediction for debugging purposes

        return render_template('result.html', results=prediction[0])

if __name__=="__main__":
    app.run(host="0.0.0.0")