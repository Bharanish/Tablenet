from flask import Flask, jsonify, request, abort, render_template, flash, redirect, url_for
from flask_jsonpify import jsonpify
import os
import cv2
import csv
import random
import pathlib
import pickle
import base64
import io
import pytesseract
from PIL import Image,ImageDraw
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from numpy.random import seed


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)
app.secret_key = "secret key for secret"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Decoder_table(tf.keras.layers.Layer):
    def __init__(self, kernel=1,  filters=[512,256,128,64,3], name="table_mask"):
        super().__init__(name=name)
        self.F1, self.F2, self.F3, self.F4, self.F5 = filters
        self.kernel = kernel
        self.conv7_table = tf.keras.layers.Conv2D(filters=self.F1, kernel_size=self.kernel, activation='relu',name = 'conv7_table')
        self.table_up_b4 = tf.keras.layers.Conv2DTranspose(filters = self.F1, kernel_size = 3, strides = 2, padding='same', name='upsamp_con1')
        self.table_up_b3 = tf.keras.layers.Conv2DTranspose(filters = self.F2, kernel_size = 3, strides = 2, padding='same', name='upsamp_con2')
        self.table_up_op_1 = tf.keras.layers.Conv2DTranspose(filters = self.F3, kernel_size = 3, strides = 2, padding='same', name='upsamp_con3')
        self.table_up_op_2 = tf.keras.layers.Conv2DTranspose(filters = self.F4, kernel_size = 3, strides = 2, padding='same', name='upsamp_con4')
        self.table_up_op_3 = tf.keras.layers.Conv2DTranspose(filters = self.F5, kernel_size = 3, strides = 2, padding='same', name='upsamp_con5')

    def call(self, X):
        inputs, pool3, pool4 = X[0], X[1], X[2]
        conv7table = self.conv7_table(inputs)
        conv7table_upb4 = self.table_up_b4(conv7table)
        pool4_append = tf.keras.layers.concatenate(inputs=[conv7table_upb4, pool4])
        conv7table_upb3 =self.table_up_b3(pool4_append)
        pool3_append = tf.keras.layers.concatenate(inputs=[conv7table_upb3, pool3])
        table_upsamp = self.table_up_op_1(pool3_append)
        table_upsamp = self.table_up_op_2(table_upsamp)
        table_mask_output = self.table_up_op_3(table_upsamp)
    
        return table_mask_output


class Decoder_column(tf.keras.layers.Layer):
    def __init__(self, kernel=1,  filters=[512,256,128,64,3], name="column_mask"):
        super().__init__(name=name)
        self.F1, self.F2, self.F3, self.F4, self.F5 = filters
        self.kernel = kernel
        self.conv7_column = tf.keras.layers.Conv2D(filters=self.F1, kernel_size=self.kernel, activation='relu',name = 'conv7_column')
        self.conv8_column = tf.keras.layers.Conv2D(filters=self.F1, kernel_size=self.kernel, activation='relu',name = 'conv8_column')
        self.table_col_up_b4 = tf.keras.layers.Conv2DTranspose(filters = self.F1, kernel_size = 3, strides = 2, padding='same', name='upsamp_con_col1')
        self.table_col_up_b3 = tf.keras.layers.Conv2DTranspose(filters = self.F2, kernel_size = 3, strides = 2, padding='same', name='upsamp_con_col2')
        self.table_col_up_op_1 = tf.keras.layers.Conv2DTranspose(filters = self.F3, kernel_size = 3, strides = 2, padding='same', name='upsamp_con_col3')
        self.table_col_up_op_2 = tf.keras.layers.Conv2DTranspose(filters = self.F4, kernel_size = 3, strides = 2, padding='same', name='upsamp_con_col4')
        self.table_col_up_op_3 = tf.keras.layers.Conv2DTranspose(filters = self.F5, kernel_size = 3, strides = 2, padding='same', name='upsamp_con_col5')
        self.drop = tf.keras.layers.Dropout(0.2)

    def call(self, X):
        inputs, pool3, pool4 = X[0], X[1], X[2]
        conv7column = self.conv7_column(inputs)
        conv7column =  self.drop(conv7column)
        conv8column = self.conv8_column(conv7column)

        conv8column_upb4 = self.table_col_up_b4(conv8column)
        pool4_append = tf.keras.layers.concatenate(inputs=[conv8column_upb4, pool4])
        conv8column_upb3 =self.table_col_up_b3(pool4_append)
        pool3_append = tf.keras.layers.concatenate(inputs=[conv8column_upb3, pool3])
        table_col_upsamp = self.table_col_up_op_1(pool3_append)
        table_col_upsamp = self.table_col_up_op_2(table_col_upsamp)
        table_col_mask_output = self.table_col_up_op_3(table_col_upsamp)
    
        return table_col_mask_output

X_input = tf.keras.layers.Input(shape=(1024, 1024, 3))
vgg_19 = tf.keras.applications.VGG19(input_tensor=X_input, include_top=False, weights='imagenet',input_shape=(1024,1024,3))
pool3 = vgg_19.get_layer('block3_pool').output
pool4 = vgg_19.get_layer('block4_pool').output
pool5 = vgg_19.get_layer('block5_pool').output

#making parameters of pre-trained vgg19 exclusing top layers to non-trainable
for layer in vgg_19.layers:
    layer.trainable = False

conv6 = tf.keras.layers.Conv2D(512, (1, 1), activation = 'relu', name='conv6_1')(pool5)
conv6 = tf.keras.layers.Dropout(0.2, name='conv6_dropout_1')(conv6)

conv6 = tf.keras.layers.Conv2D(512, (1, 1), activation = 'relu', name='conv6_2')(conv6)
conv6 = tf.keras.layers.Dropout(0.2, name='conv6_dropout_2')(conv6)

table_mask = Decoder_table(kernel=1,filters=[512,256,128,64,3])([conv6,pool3,pool4])
table_column_mask = Decoder_column(kernel=1,filters=[512,256,128,64,3])([conv6,pool3,pool4])

model = tf.keras.Model(inputs=X_input,outputs=[table_mask, table_column_mask],name="Tablenet")

BATCH_S = 4
losses = {
    "table_mask": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    "column_mask": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=losses,metrics=['accuracy'])

#loading weights of tablenet model pretrained
model.load_weights("tablenet_vgg19_e50.h5")

#this method performs the pre-processing steps before predicting the table mask for the image passed as input to the model
def test_preprocess(path_name):
    image_string = tf.io.read_file(path_name)
    image = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image  = tf.image.resize(image, [1024, 1024])
    return image

#this method returns information extracted from the image of the document containing table
def text_extraction(image,pred_table_mask,pred_column_mask):
    #original image
    doc_img = tf.keras.preprocessing.image.array_to_img(image[0])
    doc_img = cv2.cvtColor(np.asarray(doc_img), cv2.COLOR_RGB2BGR)
    doc_img = cv2.resize(doc_img, (1024,1024))
    doc_img = Image.fromarray(doc_img)
    #model predicted table mask
    tab_msk = tf.keras.preprocessing.image.array_to_img(pred_table_mask)
    
    tab_msk = tab_msk.convert('L')
    #adding grayscale table_mask image to the original image to the table portion from the image
    doc_img.putalpha(tab_msk)
    #ref: https://www.analyticsvidhya.com/blog/2020/05/build-your-own-ocr-google-tesseract-opencv/
    config = ('-l eng --oem 1 --psm 3')
    #text extraction using pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    text = pytesseract.image_to_string(doc_img, config=config)
    return text

#this method returns csv file of the information extracted from the image of the document containing table
def text_extraction_to_csv(extracted_text):
    with open(r"C:\Users\bhara\CS2\TablenetDemo\CSV_files\text_extracted.csv", 'w+', newline='') as file:
        writer=csv.writer(file)
        extracted_text = extracted_text.lstrip()
        filtered_text = []
        for ele in (extracted_text.split('\n')):
            if ele != '':
                filtered_text.append(ele.split(' '))
        for f_ele in filtered_text:
            data=','.join(f_ele)
            writer.writerow(data.split(','))

#final_func_1 takes input image and predicts and displays the table-mask, column-mask and table in the image
def final_func_1(X):
    #txt_name = X.split('/')[-1].replace('_resized.jpeg','.csv')
    files_list = tf.data.Dataset.list_files(X)
    test_size = len(list(files_list))
    test = files_list.take(test_size)
    BS = 1
    test = test.map(test_preprocess)
    test_data = test.batch(BS)
    for image in test_data.take(1):
        pred_tab, pred_col = model.predict(image)
        pred_tab = tf.argmax(pred_tab, axis=-1)
        pred_tab = pred_tab[..., tf.newaxis]
        pred_col = tf.argmax(pred_col, axis=-1)
        pred_col = pred_col[..., tf.newaxis]
        table_mask, column_mask = pred_tab[0], pred_col[0]

        #input image
        raw_inp_img = tf.keras.preprocessing.image.array_to_img(image[0])
        inp_img = cv2.cvtColor(np.asarray(raw_inp_img), cv2.COLOR_RGB2BGR)
        inp_img = cv2.resize(inp_img, (1024,1024))
        table_from_img = Image.fromarray(inp_img)
        #model predicted table mask
        pred_tab_msk = tf.keras.preprocessing.image.array_to_img(table_mask)
        pred_col_msk = tf.keras.preprocessing.image.array_to_img(column_mask)
        tab_msk = pred_tab_msk.convert('L')
        #adding grayscale table_mask image to the original image to the table portion from the image
        table_from_img.putalpha(tab_msk)
        
        extracted_text = text_extraction(image,table_mask, column_mask)
        text_extraction_to_csv(extracted_text)
        return raw_inp_img,pred_tab_msk,pred_col_msk,table_from_img

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == "POST":
        my_uploaded_file = request.files['my_uploaded_file'] # get the uploaded file
        filename = my_uploaded_file.filename
        if my_uploaded_file:
            my_uploaded_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            path_imgs ='static/uploads/*'
            raw_inp_img,pred_tab_msk,pred_col_msk,table_from_img = final_func_1(path_imgs)
            raw_inp_img.save('static/uploads/inputimage.jpeg')
            pred_tab_msk.save('static/uploads/predtabmask.jpeg')
            pred_col_msk.save('static/uploads/predcolmask.jpeg')
            table_from_img.save('static/uploads/tabinimage.png')
            inp_image = Image.open('static/uploads/inputimage.jpeg')
            tab_mask_image = Image.open('static/uploads/predtabmask.jpeg')
            col_mask_image = Image.open('static/uploads/predcolmask.jpeg')
            tab_ext_inimage = Image.open('static/uploads/tabinimage.png')
            #ref: https://buraksenol.medium.com/pass-images-to-html-without-saving-them-as-files-using-python-flask-b055f29908a
            data1 = io.BytesIO()
            data2 = io.BytesIO()
            data3 = io.BytesIO()
            data4 = io.BytesIO()
            inp_image = inp_image.resize((512,512),Image.ANTIALIAS)
            tab_mask_image = tab_mask_image.resize((512,512),Image.ANTIALIAS)
            col_mask_image = col_mask_image.resize((512,512),Image.ANTIALIAS)
            tab_ext_inimage = tab_ext_inimage.resize((512,512),Image.ANTIALIAS)
            inp_image.save(data1, "JPEG")
            tab_mask_image.save(data2, "JPEG")
            col_mask_image.save(data3, "JPEG")
            tab_ext_inimage.save(data4, "PNG")
            encoded_img_data1 = base64.b64encode(data1.getvalue())
            encoded_img_data2 = base64.b64encode(data2.getvalue())
            encoded_img_data3 = base64.b64encode(data3.getvalue())
            encoded_img_data4 = base64.b64encode(data4.getvalue())
            return render_template('index.html', img_data1=encoded_img_data1.decode('utf-8'),img_data2=encoded_img_data2.decode('utf-8'),
            img_data3=encoded_img_data3.decode('utf-8'),img_data4=encoded_img_data4.decode('utf-8'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080,debug=True)