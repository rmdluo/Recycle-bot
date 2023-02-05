import os

from flask import Flask
from flask import render_template
from flask import flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from predict import *

CLASSIFIER = Classifier("model_96.tflite", "classes.txt")

 
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
@app.route('/')
def hello(): 
    return render_template('file_upload.html')

@app.route('/upload', methods=["POST"])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(UPLOAD_FOLDER, filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		text = CLASSIFIER.string_predict(os.path.join(UPLOAD_FOLDER, filename))
		return render_template('file_upload.html', filename=filename, text=text)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)
	
@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__": 
    app.run();