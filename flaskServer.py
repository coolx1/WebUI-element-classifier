# import os
from flask import Flask, flash ,request, redirect, url_for, render_template
# from werkzeug.utils import secure_filename
import cv2
import json
import numpy as np
# from keras.preprocessing import image as i
# from keras.models import model_from_json
from PIL import Image
import io
import running as r

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)

def create_json(file):
    dict ={}
    dict["class"] = r.predict(file)
    if(isinstance(file, str)):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.cvtColor(file, cv2.COLOR_RGB2GRAY)
    w,h = img.shape
    dict["height"]= w
    dict["width"] = h
    arr = [0] * 2
    for i in range(w):
        for j in range(h):
            k = img[i, j]
            if(k<= 127):
                arr[0]+=1
            else:
                arr[1]+=1.0
    if(arr[1]>0):
        ratio = arr[0] / arr[1]
    else:
        ratio = None
    dict["ratio_B/W"] = ratio
    json_string = json.dumps(dict)
    return json_string


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

stri = ""
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global stri
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            img = file.read()
            image = Image.open(io.BytesIO(img))
            image = np.array(image)
            stri = create_json(image)
            return redirect(url_for('upload_file'))
    return render_template("upload.html",strin = stri)


if __name__ == "__main__":
    app.run(debug=True)
