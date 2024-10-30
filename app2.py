import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import json

# Cargar modelo ResNet50 preentrenado
model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

# Ruta del dataset
DATASET_PATH = "./MexicanFoodDataSet-master"

# Función para extraer embeddings de las imágenes
def extract_image_features(img_path):
    try:
        img = Image.open(img_path).resize((224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array).flatten()
        return features
    except Exception as e:
        print(f"Error al procesar la imagen {img_path}: {e}")
        return None

# Función para cargar las imágenes y sus datos
def load_image_data(base_path):
    features, image_paths, calorias_data, ingredients_data = [], [], [], []

    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            img_file = next((f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))), None)
            if img_file:
                img_path = os.path.join(folder_path, img_file)
                feature = extract_image_features(img_path)

                if feature is not None:
                    features.append(feature)
                    image_paths.append(img_path)

                    # Cargar JSON correspondiente
                    json_file = next((f for f in os.listdir(folder_path) if f.endswith('.json')), None)
                    if json_file:
                        with open(os.path.join(folder_path, json_file), 'r') as f:
                            dish_data = json.load(f)
                        dish_name = list(dish_data.keys())[0]
                        calorias = sum(float(item.get('Calorias', 0)) for item in dish_data[dish_name])
                        ingredients = ", ".join([item['Ingrediente'] for item in dish_data[dish_name]])

                        calorias_data.append(calorias)
                        ingredients_data.append(ingredients)

    return np.array(features), image_paths, calorias_data, ingredients_data

# Cargar datos
features, image_paths, calorias_data, ingredients_data = load_image_data(DATASET_PATH)

# Selección entre K-means o t-SNE
reduction_method = st.radio("Selecciona el método de reducción dimensional:", ["PCA", "t-SNE"])

if reduction_method == "PCA":
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    st.write(f"Varianza explicada por PCA: {pca.explained_variance_ratio_}")
else:
    tsne = TSNE(n_components=2, random_state=42)
    reduced_features = tsne.fit_transform(features)

# Aplicar K-means para clusterización
num_clusters = st.slider("Selecciona el número de clústeres:", 2, 10, 5)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(features)

# Crear DataFrame con los resultados
df = pd.DataFrame(reduced_features, columns=['Dim1', 'Dim2'])
df['Cluster'] = labels
df['Calorias'] = calorias_data
df['Ingredientes'] = ingredients_data
df['Image'] = image_paths

# Visualización con Plotly
fig = px.scatter(
    df, x='Dim1', y='Dim2', color='Cluster',
    hover_data=['Ingredientes', 'Calorias'], 
    title=f"Visualización de {reduction_method} con K-means"
)
st.plotly_chart(fig)

# Selección del clúster con radio button
selected_cluster = st.radio(
    "Selecciona un clúster para ver las imágenes:", 
    options=df['Cluster'].unique()
)

# Filtrar las imágenes del clúster seleccionado
filtered_df = df[df['Cluster'] == selected_cluster]

# Mostrar las imágenes en un grid
st.write(f"Imágenes en el clúster {selected_cluster}:")
num_cols = 5
rows = (len(filtered_df) + num_cols - 1) // num_cols

for i in range(rows):
    cols = st.columns(num_cols)
    for j, col in enumerate(cols):
        index = i * num_cols + j
        if index < len(filtered_df):
            img_path = filtered_df.iloc[index]['Image']
            col.image(img_path, use_column_width=True)
