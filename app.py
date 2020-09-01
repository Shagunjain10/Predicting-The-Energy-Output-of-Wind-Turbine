import numpy as np

from flask import Flask, request, jsonify, render_template
import pickle



app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict',methods=['POST'])
def predict():
	#print(request.form.values())
	#print("\n\n\n")
	int_features = [int(x) for x in request.form.values()]
	#[int(x) for x in request.form.values()]
	#print(int_features)
	#final_features = np.array(int_features)
	#print(final_features)
	int_features = np.reshape(int_features, (1,3))
	prediction = model.predict(int_features)
	if(prediction<0):
		prediction=0
	output = round(prediction[0], 2)
	print(output)
	return render_template('main.html', prediction_text='Generated Energy from wind turbine is {} KWh'.format(output))


if __name__ == "__main__":
    app.run(debug=True)