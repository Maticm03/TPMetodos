import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Cargar imágenes y convertirlas en vectores
def load_and_convert_images():
    image_vectors = []
    for i in range(16):
        image_path = f"C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img{'0' if i < 10 else ''}{i}.jpeg"
        image = Image.open(image_path).convert('L')
        image_vector = np.array(image).flatten()
        image_vectors.append(image_vector)
    return np.array(image_vectors)

# Realizar SVD a la matriz de datos
def apply_svd(data_matrix):
    U, s, Vt = np.linalg.svd(data_matrix, full_matrices=False)
    return U, s, Vt

# Comprimir y reconstruir las imágenes
def compress_and_reconstruct_images(U, s, Vt, k):
    compressed_images = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return compressed_images

# Calcular la similaridad entre pares de imágenes
def calculate_similarity(images):
    similarity_values = []  # Almacenar las similaridades entre pares de imágenes
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            img1 = images[i].flatten()  # Aplana la primera imagen
            img2 = images[j].flatten()  # Aplana la segunda imagen
            similarity = np.linalg.norm(img1 - img2)  # Calcular la distancia euclidiana
            similarity_values.append(similarity)
    return similarity_values

# Mostrar imágenes originales y reconstrucciones
def show_images_and_reconstructions(images, reconstructions_first, reconstructions_last, similarities):
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
        ax.set_title(f'SVD Primeras {i}')
        ax.axis('off')
    plt.suptitle('Reconstrucciones con Primeras Dimensiones')
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(reconstructions_last[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'SVD Últimas {i}')
        ax.axis('off')
    plt.suptitle('Reconstrucciones con Últimas Dimensiones')
    plt.show()

    # Visualizar la similaridad entre pares de imágenes en un gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(similarities)), similarities, marker='o', linestyle='-')
    plt.title('Similaridad entre pares de imágenes', fontsize=12)
    plt.xlabel('Pares de imágenes')
    plt.ylabel('Similaridad (Distancia Euclidiana)')
    plt.grid(True)
    plt.show()

# Main
image_vectors = load_and_convert_images()
U, s, Vt = apply_svd(image_vectors)
k_first = [5, 10, 20]  # Tres valores distintos para las primeras dimensiones
k_last = [5, 10, 20]  # Tres valores distintos para las últimas dimensiones

reconstructions_first = [compress_and_reconstruct_images(U, s, Vt, k) for k in k_first]
reconstructions_last = [compress_and_reconstruct_images(U, s, Vt, -k) for k in k_last]  # Usar el negativo para seleccionar las últimas dimensiones

similarities = calculate_similarity(image_vectors)

show_images_and_reconstructions(image_vectors, reconstructions_first, reconstructions_last, similarities)
