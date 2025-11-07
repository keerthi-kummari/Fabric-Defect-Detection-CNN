from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = tf.keras.models.load_model('fabric_defect_model.h5')

@app.route('/')
def home():
    return "Fabric Defect Detection Web App"

if __name__ == '__main__':
    app.run(debug=True)
