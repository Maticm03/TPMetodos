import numpy as np
from scipy.spatial.distance import euclidean
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# Lista de valores de d a considerar
d_values = [1, 5, 10, 20]  # Puedes ajustar estos valores

similarities_by_d = {d: [] for d in d_values}

for d in d_values:
    reconstructed_images = []  # Crear una lista para almacenar las imágenes reconstruidas

    for image_file_name in image_file_names:
        try:
            # Cargar la imagen
            image = mpimg.imread(image_file_name)

            # Aplicar SVD a la matriz de la imagen
            U, S, VT = np.linalg.svd(image, full_matrices=False)

            # Reconstruir la imagen original utilizando las primeras dimensiones
            reconstructed_matrix = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
            reconstructed_images.append(reconstructed_matrix)

        except Exception as e:
            print(f"Error al procesar la imagen: {image_file_name}. Detalles del error: {e}")

    # Calcular la similaridad entre pares de imágenes para el valor de d actual utilizando la distancia euclidiana
    similarities = []  # Almacenar las similaridades entre pares de imágenes para d
    for pair in combinations(range(len(image_file_names)), 2):
        img1 = reconstructed_images[pair[0]].flatten()  # Aplana la primera imagen
        img2 = reconstructed_images[pair[1]].flatten()  # Aplana la segunda imagen
        similarity = euclidean(img1, img2)
        similarities.append(similarity)

    similarities_by_d[d] = similarities  # Almacenar las similaridades para este valor de d


# Visualizar la similaridad entre pares de imágenes para cada valor de d en un gráfico
plt.figure(figsize=(10, 6))
for d in d_values:
    plt.plot(range(len(similarities_by_d[d])), similarities_by_d[d], marker='o', linestyle='-', label=f'd={d}')

plt.title('Similaridad entre pares de imágenes para diferentes valores de d', fontsize=12)
plt.xlabel('Pares de imágenes')
plt.ylabel('Similaridad (Distancia Euclidiana)')
plt.legend()
plt.grid(True)
plt.show()
