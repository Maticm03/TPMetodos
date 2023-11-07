import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA

# Cargar imágenes y convertirlas en vectores
def load_and_convert_images():
    image_vectors = []
    for i in range(16):
        image_path = f"C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img{'0' if i < 10 else ''}{i}.jpeg"
        image = Image.open(image_path).convert('L')
        image_vector = np.array(image).flatten()
        image_vectors.append(image_vector)
    return np.array(image_vectors)

# Realizar PCA a la matriz de datos
def apply_pca(data_matrix, n_components):
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_matrix)
    data_reconstructed = pca.inverse_transform(data_pca)
    return data_reconstructed

# Mostrar imágenes originales y reconstrucciones
def show_images_and_reconstructions(images, reconstructions_first, reconstructions_last):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Imagen {i}')
        ax.axis('off')
    plt.suptitle('Imágenes Originales')
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(reconstructions_first[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'PCA Primeras {i}')
        ax.axis('off')
    plt.suptitle('Reconstrucciones con PCA (Primeras Dimensiones)')
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(reconstructions_last[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'PCA Últimas {i}')
        ax.axis('off')
    plt.suptitle('Reconstrucciones con PCA (Últimas Dimensiones)')
    plt.show()

# Main
image_vectors = load_and_convert_images()
n_components_first = 10  # Número de componentes principales para las primeras dimensiones
n_components_last = 10 # Número de componentes principales para las últimas dimensiones

reconstructions_first = apply_pca(image_vectors, n_components_first)
reconstructions_last = apply_pca(image_vectors, n_components_last)

show_images_and_reconstructions(image_vectors, reconstructions_first, reconstructions_last)

# Calcular la similaridad entre pares de imágenes para diferentes valores de d y almacenar los resultados
n_components_values = [1, 5, 10, 16]  # Valores de n_components a considerar
similarity_values = []  # Almacenar las similaridades para cada valor de n_components

image_vectors = load_and_convert_images()

for n_components in n_components_values:
    # Comprimir y reconstruir las imágenes con el valor de n_components actual
    reconstructions = apply_pca(image_vectors, n_components)

    # Calcular la similaridad entre pares de imágenes para n_components utilizando la distancia euclidiana
    similarities = []  # Almacenar las similaridades entre pares de imágenes para n_components
    for i in range(len(image_vectors)):
        for j in range(i + 1, len(image_vectors)):
            img1 = reconstructions[i].flatten()  # Aplana la primera imagen
            img2 = reconstructions[j].flatten()  # Aplana la segunda imagen
            similarity = np.linalg.norm(img1 - img2)  # Calcular la distancia euclidiana
            similarities.append(similarity)

    similarity_values.append(similarities)

# Visualizar la similaridad entre pares de imágenes para cada valor de n_components en un gráfico
plt.figure(figsize=(10, 6))
for i, n_components in enumerate(n_components_values):
    plt.plot(range(len(similarity_values[i])), similarity_values[i], marker='o', linestyle='-', label=f'n_components={n_components}')

plt.title('Similaridad entre pares de imágenes para diferentes valores de n_components', fontsize=12)
plt.xlabel('Pares de imágenes')
plt.ylabel('Similaridad (Distancia Euclidiana)')
plt.legend()
plt.grid(True)
plt.show()