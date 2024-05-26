from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import Dense, Flatten, Input

app = Flask(__name__)

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define additional layers for feature extraction
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
output = Dense(128)(x)

# Create feature extraction model
feature_extractor = Model(inputs=base_model.input, outputs=output)

def image_stitching(image1, image2):
    # Resize images
    image1_resized = cv2.resize(image1, (224, 224))
    image2_resized = cv2.resize(image2, (224, 224))

    # Extract features using the feature extractor model
    features1 = feature_extractor.predict(np.expand_dims(image1_resized, axis=0))
    features2 = feature_extractor.predict(np.expand_dims(image2_resized, axis=0))

    # Calculate cosine similarity between feature maps
    similarity_matrix = cosine_similarity(features1.T, features2.T)

    # Find the best matching points
    src_pts = []
    dst_pts = []
    for i in range(similarity_matrix.shape[0]):
        j = np.argmax(similarity_matrix[i])
        src_pts.append((i % 14 * 16, i // 14 * 16))
        dst_pts.append((j % 14 * 16, j // 14 * 16))

    # Convert to numpy arrays
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    stitched_width = image1_resized.shape[1] + image2_resized.shape[1]
    stitched_height = max(image1_resized.shape[0], image2_resized.shape[0])

    # Create an empty canvas for the stitched image
    stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # Combine the images onto the stitched canvas
    stitched_image[0:image1_resized.shape[0], 0:image1_resized.shape[1]] = image1_resized
    stitched_image[0:image2_resized.shape[0], image1_resized.shape[1]:] = image2_resized

    return stitched_image

@app.route('/stitch', methods=['POST'])
def stitch_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return 'Error: Please provide both image1 and image2.'

    image1 = cv2.imdecode(np.frombuffer(request.files['image1'].read(), np.uint8), cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(np.frombuffer(request.files['image2'].read(), np.uint8), cv2.IMREAD_COLOR)

    stitched_image = image_stitching(image1, image2)

    # Convert stitched image to JPEG format
    _, stitched_buffer = cv2.imencode('.jpg', stitched_image)
    stitched_data = stitched_buffer.tobytes()

    return stitched_data, 200, {'Content-Type': 'image/jpeg'}

if __name__ == '_main_':
    app.run(debug=True)
