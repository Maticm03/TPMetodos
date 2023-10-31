import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.linalg import svd
from itertools import combinations
from scipy.spatial.distance import euclidean

image_file_names = ["C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img00.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img01.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img02.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img03.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img04.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img05.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img06.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img07.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img08.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img09.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img10.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img11.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img12.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img13.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img14.jpeg",
                    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img15.jpeg"]

# Número de componentes singulares a mantener
n_componentes_primero = 10  # Puedes ajustar este valor para las primeras dimensiones
n_componentes_ultimo = 10  # Puedes ajustar este valor para las últimas dimensiones

# Crear una lista para almacenar las imágenes reconstruidas
reconstructed_images_primero = []
reconstructed_images_ultimo = []

# Crear una lista para almacenar las similaridades entre pares de imágenes
similarities_primero = []
similarities_ultimo = []

for image_file_name in image_file_names:
    try:
        # Cargar la imagen
        image = mpimg.imread(image_file_name)

        # Aplicar SVD a la matriz de la imagen
        U, S, VT = np.linalg.svd(image, full_matrices=False)

        # Reconstruir la imagen original utilizando las primeras dimensiones
        reconstructed_matrix_primero = np.dot(U[:, :n_componentes_primero], np.dot(np.diag(S[:n_componentes_primero]), VT[:n_componentes_primero, :]))
        reconstructed_images_primero.append(reconstructed_matrix_primero)

        # Reconstruir la imagen original utilizando las últimas dimensiones
        reconstructed_matrix_ultimo = np.dot(U[:, -n_componentes_ultimo:], np.dot(np.diag(S[-n_componentes_ultimo:]), VT[-n_componentes_ultimo:, :]))
        reconstructed_images_ultimo.append(reconstructed_matrix_ultimo)

    except Exception as e:
        print(f"Error al procesar la imagen: {image_file_name}. Detalles del error: {e}")

# Calcular y mostrar la similaridad entre pares de imágenes para las primeras dimensiones
for pair in combinations(range(len(image_file_names)), 2):
    img1 = reconstructed_images_primero[pair[0]].flatten()
    img2 = reconstructed_images_primero[pair[1]].flatten()
    similarity = euclidean(img1, img2)
    similarities_primero.append(similarity)

# Calcular y mostrar la similaridad entre pares de imágenes para las últimas dimensiones
for pair in combinations(range(len(image_file_names)), 2):
    img1 = reconstructed_images_ultimo[pair[0]].flatten()
    img2 = reconstructed_images_ultimo[pair[1]].flatten()
    similarity = euclidean(img1, img2)
    similarities_ultimo.append(similarity)

# Crear una figura para mostrar las similaridades
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(similarities_primero, marker='o')
plt.title('Similaridades con las primeras dimensiones', fontsize=12)
plt.xlabel('Pares de imágenes')
plt.ylabel('Distancia Euclidiana')

plt.subplot(1, 2, 2)
plt.plot(similarities_ultimo, marker='o')
plt.title('Similaridades con las últimas dimensiones', fontsize=12)
plt.xlabel('Pares de imágenes')
plt.ylabel('Distancia Euclidiana')

plt.tight_layout()
plt.show()
