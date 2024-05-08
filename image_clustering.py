# for loading/processing the images
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "flower_images"
os.chdir(path)

flowers = []

with os.scandir(path) as files:
    for file in files:
        if file.name.endswith('.png'):
            flowers.append(file.name)
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    file = "/home/user/Desktop/ML/flower_images/flower_images/" + file
    img = load_img(file, target_size = (224,224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx)

    return features
   
data = {}
p = r"flower_features.pk1"

for flower in flowers:
    feat = extract_features(flower, model)
    data[flower] = feat
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1, 4096)

# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('/home/user/Desktop/ML/flower_images/flower_images/flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components = 100, random_state = 22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters = len(unique_labels), random_state = 22)
kmeans.fit(x)

groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
      
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    files = groups[cluster]
    if len(files) > 30:
        print("Clipping cluster size from", {len(files)}, "to 30")
        files = files[:29]

    for index, file in enumerate(files):
        file = "/home/user/Desktop/ML/flower_images/flower_images/" + file

        plt.subplot(10, 10, index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

for cluster_id in groups:
    view_cluster(cluster_id)

plt.show()