import os
import io
import numpy as np

import keras
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import (VGG19, preprocess_input) # has been added
# from keras.applications.xception import (
#     Xception, preprocess_input, decode_predictions)
from keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify, render_template

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# model = None
model_vgg19 = None # has been added
model_clothing = None # has been added
graph = None

# 3 lines below ahs been added
clothing_dict = {0: "pants", 1: "purse", 2:"skirt", 3:"sneaker", 4:"sweater", 5:"t-shirt"}
def decode_predictions(preds):
    return clothing_dict[np.argmax(preds)]

def load_models():
    # global model
    global model_vgg19 # has been added
    global model_clothing # has been added
    global graph
    # model = Xception(weights="imagenet")
    model_vgg19 = VGG19(weights="imagenet") # has been added
    model_clothing = keras.models.load_model("ml-clothing-class.h5") # has been added
    graph = K.get_session().graph


load_models()


def prepare_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    data = {"success": False}
    if request.method == 'POST':
        if request.files.get('file'):
            # read the file
            file = request.files['file']

            # read the filename
            filename = file.filename

            # create a path to the uploads folder
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)
            data["file"] = filepath

            # Load the saved image using Keras and resize it to the Xception
            # format of 299x299 pixels
            image_size = (224, 224)
            im = keras.preprocessing.image.load_img(filepath,
                                                    target_size=image_size,
                                                    grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)

            global graph
            with graph.as_default():
                preds_vgg19 = model_vgg19.predict(image) # has been added
                preds = model_clothing.predict(preds_vgg19) # has been added

                # results = decode_predictions(preds)
                # print the results

                # print(results)

                data["prediction"] = decode_predictions(preds)

                # # loop over the results and add them to the list of
                # # returned predictions
                # for (imagenetID, label, prob) in results[0]:
                #     r = {"label": label, "probability": float(prob)}
                #     data["predictions"].append(r)

                # # indicate that the request was a success
                data["success"] = True

        # return jsonify(data)
        return render_template("results.html", data=data)

    return render_template("index.html", data=data)

    # return
    # '''
    #     <!doctype html>
    #     <title>Upload new File</title>
    #     <h1>Upload new File</h1>
    #     <form method=post enctype=multipart/form-data>
    #     <p><input type=file name=file>
    #         <input type=submit value=Upload>
    #     </form>
    # '''

if __name__ == "__main__":
    app.run(debug=True)
