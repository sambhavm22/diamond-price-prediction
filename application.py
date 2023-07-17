from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline


application= Flask(__name__)

app= application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data= CustomData(
            carat= float(request.form.get('carat')),
            depth= float(request.form.get('depth')),
            table= float(request.form.get('table')),
            x= float(request.form.get('x')),
            y= float(request.form.get('y')),
            z= float(request.form.get('z')),
            cut= request.form.get('cut'),
            color= request.form.get('color'),
            clarity= request.form.get('clarity')
        )
        final_new_data= data.get_data_as_data_frame()
        print(final_new_data)
        predict_pipeline= PredictionPipeline()
        pred= predict_pipeline.prediction_data(final_new_data)

        results = round(pred[0], 2)
        print(results)

        return render_template('home.html', final_result= results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port= 5002, debug=True)