import os
import pickle

import numpy as np
import requests
from flask import Flask, jsonify, make_response, request
from numpy.linalg import norm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='max')
pca: PCA = pickle.load(open('./pca.pickle', 'rb'))
neighbors: NearestNeighbors = pickle.load(open('./neighbors.pickle', 'rb'))
filenames = pickle.load(open('./filenames.pickle', 'rb'))

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "dmlzdWFsLWluZm9ybWF0aW9uLXJldHJpZXZhbA=="


def extract_features(img_path, model):
    input_shape = (224, 224, 3)
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / norm(flattened_features)
    return normalized_features
    

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_preflight_response()

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return _corsify_actual_response(jsonify({
                "msg": "No file part"
            }))

        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return _corsify_actual_response(jsonify({
                "msg": "No selected file"
            }))

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            feature = extract_features(filepath, model=model)
            feature_compressed = pca.transform([feature])[0]
            
            if "type" in request.args and request.args.get("type") == "es":
                return es_request(feature_compressed=feature_compressed)
            
            return in_memory_request(feature_compressed=feature_compressed)

    return _corsify_actual_response(jsonify({
        "msg": "Invalid request"
    }))
            
def in_memory_request(feature_compressed):
    distances, indices = neighbors.kneighbors([feature_compressed])
    similar_image_paths = [filenames[indices[0][i]] for i in range(0, 10)]
    
    return _corsify_actual_response(jsonify([{
        "path": w,
        "distance": distances[0][i]
        } for i, w in enumerate(similar_image_paths)]))


def es_request(feature_compressed):
    return _corsify_actual_response(jsonify(requests.post("http://45.76.188.206:9200/visual-information-retrieval-v2/_search?pretty", json={
                "knn": {
                    "field": "image-vector",
                    "query_vector": [float(w) for w in feature_compressed],
                    "k": 10,
                    "num_candidates": 100
                }, 
                "_source": {
                    "excludes": [
                        "image-vector"
                    ]
                }
            }).json()))