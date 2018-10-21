import segmentation as seg
import cv2
import numpy as np
from keras.preprocessing import image as i
from keras.models import model_from_json
import keras


def preprocess(s):
    array = seg.segment(s)
    if(len(array)<1):
        keras.backend.clear_session()
        return "Not segmentable!"
    image = array[0]
    image = i.img_to_array(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = np.expand_dims(image, axis=0)
    return image

def imageClass(clas):
    if(clas == 0):
        return "cards"
    elif(clas == 1):
        return "checkbox"
    elif(clas == 2):
        return "combobox"
    elif(clas == 3):
        return "enable_disable"
    elif(clas == 4):
        return "forward"
    elif(clas == 5):
        return "home"
    elif(clas == 6):
        return "information_icon"
    elif(clas == 7):
        return "tabview"
    elif(clas == 8):
        return "textbox"
    elif (clas == 9):
        return "window_header"
    elif (clas == 10):
        return "zoom"
    else:
        return "not_identified"

def predict(filename):
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    image = preprocess(filename)
    if (isinstance(image, str)):
        return image
    result = loaded_model.predict(image)
    keras.backend.clear_session()
    return imageClass(np.argmax(result[0]))

if __name__ == "__main__":
    # loading the JSON object
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    image = preprocess("sample3.png")
    result = loaded_model.predict(image)
    print(imageClass(np.argmax(result[0])))




