import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.applications.vgg16 import VGG16 

# Function to extract features from a frame
def extract_features(frame, model):
    frame = cv2.resize(frame, (224, 224))
    imgx = np.expand_dims(frame, axis = 0)
    features = model.predict(imgx)
    return features.flatten()

# Load pre-trained VGG16 model
model = VGG16(weights = 'imagenet', include_top = False)

# Initialize KMeans clustering
kmeans = KMeans(n_clusters = 5, random_state = 42)

# Read the video
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

frames = []
frame_indices = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_indices.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

# Extract features from each frame
features = []
for frame in frames:
    features.append(extract_features(frame, model))

features = np.array(features)

# Reduce dimensionality using PCA
pca = PCA(n_components = 100, random_state = 22)
features_pca = pca.fit_transform(features)

# Cluster the features
kmeans.fit(features_pca)

# Group frames into scenes
scenes = {}
for i, label in enumerate(kmeans.labels_):
    if label not in scenes:
        scenes[label] = []
    scenes[label].append(frame_indices[i])  # Use frame indices instead of index

# Display frames from each scene in order
for scene_id in sorted(scenes.keys()):
    print(f"Scene {scene_id + 1}:")
    for index in scenes[scene_id]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)  # Set the frame position
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Scene {scene_id + 1}", frame)
        cv2.waitKey(50)  # Adjust the delay as needed
    cv2.destroyWindow(f"Scene {scene_id + 1}")

cap.release()
cv2.destroyAllWindows()