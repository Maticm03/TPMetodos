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
        ax.set_title(f'Imagen {i}')
        ax.axis('off')
    plt.suptitle('Reconstrucciones con Primeras Dimensiones')
    plt.show()

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(reconstructions_last[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Imagen {i}')
        ax.axis('off')
    plt.suptitle('Reconstrucciones con Últimas Dimensiones')
    plt.show()

# Main
image_vectors = load_and_convert_images()
U, s, Vt = apply_svd(image_vectors)
k_first = 5  # Puedes ajustar este valor para las primeras dimensiones
k_last = 5 # Puedes ajustar este valor para las últimas dimensiones
reconstructions_first = compress_and_reconstruct_images(U, s, Vt, k_first)
reconstructions_last = compress_and_reconstruct_images(U, s, Vt, -k_last)  # Usar el negativo para seleccionar las últimas dimensiones

show_images_and_reconstructions(image_vectors, reconstructions_first, reconstructions_last)

# Calcular la similaridad entre pares de imágenes para diferentes valores de d y almacenar los resultados
d_values = [1, 5, 10, 16, 20]  # Valores de d a considerar
similarity_values = []  # Almacenar las similaridades para cada valor de d

for d in d_values:
    # Comprimir y reconstruir las imágenes con el valor de d actual
    reconstructions_d = compress_and_reconstruct_images(U, s, Vt, d)

    # Calcular la similaridad entre pares de imágenes para d utilizando la distancia euclidiana
    similarities_d = []  # Almacenar las similaridades entre pares de imágenes para d
    for i in range(len(image_vectors)):
        for j in range(i + 1, len(image_vectors)):
            img1 = reconstructions_d[i].flatten()  # Aplana la primera imagen
            img2 = reconstructions_d[j].flatten()  # Aplana la segunda imagen
            similarity = np.linalg.norm(img1 - img2)  # Calcular la distancia euclidiana
            similarities_d.append(similarity)

    similarity_values.append(similarities_d)

# Visualizar la similaridad entre pares de imágenes para cada valor de d en un gráfico
plt.figure(figsize=(10, 6))
for i, d in enumerate(d_values):
    plt.plot(range(len(similarity_values[i])), similarity_values[i], marker='o', linestyle='-', label=f'd={d}')

plt.title('Similaridad entre pares de imágenes para diferentes valores de d', fontsize=12)
plt.xlabel('Pares de imágenes')
plt.ylabel('Similaridad (Distancia Euclidiana)')
plt.legend()
plt.grid(True)
plt.show()
