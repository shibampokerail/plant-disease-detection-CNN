from flask import Flask, flash, render_template, request, redirect
from predict import predict_disease
# from werkzeug.utils import secure_filename

import os
import time

FILE_NAME = ""

app = Flask(__name__)

DEVELOPMENT = 'dev'
PRODUCTION = 'prod'
HOST_IP = ''
ENV = DEVELOPMENT

# print("connected to the database")
# print("unable to connect to the database")

if ENV == DEVELOPMENT:
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:12345@localhost/share_db'
    HOST_IP = '' # type ipconfig in cmd and put the IP of ipv4 from wireless section
    else:
    # remote appdatabase uri
    app.config['SQLALCHEMY_DATABASE_URI'] = ''
    app.debug = False

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

app.config['UPLOAD_FOLDER'] = 'static'
if not os.path.exists('static'):
    os.mkdir('static')

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_PATH'] = 30 * 1000 * 1000  # 30mb

CATEGORIES = ["apple_blackrot", "corn_common_rust", "potato_early_blight", "apple_scab", "corn_greyleafspot",
              "potato_lateblight", "apple_healthy", "corn_healthy", "potato_healthy"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def file_upload(filerequest, name: str):
    if 'file' not in filerequest.files:
        flash('No file part')
        return redirect("/?f_name=nofile&plant=no_data_supplied&disease=no_data_supplied")


    # print(filerequest.files)
    file = filerequest.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(filerequest.url)
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = name + "." + file.filename.rsplit('.', 1)[1].lower()
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        prediction_array = predict_disease(f"static/{filename}")
        # the output will be in the form of [0.,0.,0.,1,0.,0.,0.,0.] and the one with the one is the prediction
        prediction = CATEGORIES[prediction_array.index(1)]
        return redirect(f'/?f_name={filename}&prediction="{prediction}"', )


@app.route('/')
def index():
    if request.args.get("f_name"):
        plant = request.args.get("prediction").split("_")[0].replace('"', "")
        disease = ("".join(request.args.get("prediction").split("_")[1:])).replace('"', "")
        print(plant + disease)

        return render_template("index.html", image=request.args.get("f_name"),
                               plant=plant, disease=disease)
    return render_template("index.html")


@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == 'POST':
        # check if the post request has the file part
        name = request.files["file"].filename.split(".")[0]
        return file_upload(request, name)


if __name__ == "__main__":
    try:
        pass
        app.run(host=HOST_IP, port=8000)
    except:
        app.run()
