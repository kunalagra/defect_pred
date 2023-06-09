from flask import Flask, render_template, request,url_for
import tensorflow as tf
from tensorflow.keras.layers import *
import os
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)
app.secret_key = "SessionKEy145"
app.config['SESSION_TYPE'] = 'filesystem'

img_height, img_width = 300, 300
input_shape = (img_height, img_width, 3)

class CNNBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, pool_size, dropout_rate):
        super(CNNBlock, self).__init__()
        self.C1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        self.P1 = MaxPooling2D(pool_size=pool_size, strides=2, padding=padding)
        self.Dr1 = Dropout(dropout_rate)
        
    def call(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        x = self.P1(x)
        y = self.Dr1(x)
        return y

class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate):
        super(DenseBlock, self).__init__()
        self.D1 = Dense(units, activation='relu')
        self.B1 = BatchNormalization()
        self.D2 = Dense(units * 2, activation='relu')
        self.D3 = Dense(units * 2, activation='relu')
        self.D4 = Dense(units, activation='relu')
        self.Dr1 = Dropout(dropout_rate)
        self.D5 = Dense(1, activation='sigmoid')
        
    def call(self, x):
        x = self.D1(x)
        x = self.B1(x)
        x = self.D2(x)
        x = self.D3(x)
        x = self.D4(x)
        x = self.Dr1(x)
        y = self.D5(x)
        return y

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.C1 = Conv2D(filters=32, kernel_size=(3 * 3), strides=1, padding='same', input_shape=input_shape)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        
        self.layer1 = CNNBlock(filters=32, kernel_size=(3 * 3), strides=1, padding='same', pool_size=(2 * 2), dropout_rate=0.3)
        self.layer2 = CNNBlock(filters=64, kernel_size=(3 * 3), strides=1, padding='same', pool_size=(2 * 2), dropout_rate=0.4)
        self.layer3 = CNNBlock(filters=32, kernel_size=(3 * 3), strides=1, padding='same', pool_size=(2 * 2), dropout_rate=0.3)
        
        self.F1 = Flatten()
        self.layer4 = DenseBlock(units=64, dropout_rate=0.3)
        
    def call(self, x):
        x = self.C1(x)
        x = self.B1(x)
        x = self.A1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.F1(x)
        
        y = self.layer4(x)
        return y
    
    def __repr__(self):
        name = 'Bangle_Net'
        return name

    def build_graph(self):
        x = tf.keras.Input(shape=(300,300,3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

model = NeuralNetwork()


@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    f = request.files['upload']
    fp = os.path.join("static", f.filename)
    f.save(fp)
    img = tf.keras.preprocessing.image.load_img(
    fp, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = img_array / 255
    img_array = tf.expand_dims(img_array, 0)

    res = model.predict(img_array)

    if res[0][0] < 0.5:
      ans = f"""
        <div class="row text-center mt-5">
            <h1 class="display-1 text-danger mb-5"><b>REJECT</b></h1>
            <div class="card mx-auto" style="width: 90vw; background-color: antiquewhite;">
                <div class="card-header">
                    Product Type: <b>Bangle</b>
                </div>

                <div class="col-lg-6 mx-auto">
                <img class="card-img-top rounded" src="{ url_for('static', filename=f.filename)}" alt="Card image cap">
                </div>
                <div class="card-body">
                    <h5 class="card-title">Product Code: <b>4oSEGXNu</b></h5>
                    <p class="card-text">This product, on the Flat Belt Conveyor Belt on production line 2, was found to
                        be defective and has been rejected from the production line.
                    </p>
                    <p class="card-text">The possible defects could be:</p>
                    <ul class="list-group">
                        <li class="list-group-item">Dimensional or tolerance defect</li>
                        <li class="list-group-item">Material defect</li>
                        <li class="list-group-item">Surface defect</li>
                        <li class="list-group-item">Assembly defect</li>
                    </ul>
                </div>
            </div>
        </div>
      """
    else:
      ans = f"""
        <div class="row text-center mt-5">
            <h1 class="display-1 text-success mb-5"><b>ACCEPT</b></h1>
            <div class="card mx-auto" style="width: 90vw; background-color: antiquewhite;">
                <div class="card-header">
                    Product Type: <b>Bangle</b>
                </div>
                <div class="col-lg-6 mx-auto">
                <img class="card-img-top rounded" src="{ url_for('static', filename=f.filename)}" alt="Card image cap">
                </div>
                <div class="card-body">
                    <h5 class="card-title">Product Code: <b>VGvmGPEr</b></h5>
                    <p class="card-text">This product, on the Flat Belt Conveyor Belt on production line 2, meets the required quality standards and is ready to deliver.
                    </p>

                </div>
            </div>
        </div>
      """
    return render_template('output.html', output = ans)

if __name__ == "__main__":
    app.run()