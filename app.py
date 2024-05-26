from flask import Flask, render_template, request
from predict import predict_image

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            prediction = predict_image(image_file)
            return render_template('result.html', prediction=prediction)
        else:
            return render_template('index.html', error='Please upload an image')

if __name__ == '__main__':
    app.run(debug=True)
