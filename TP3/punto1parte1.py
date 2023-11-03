import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from scipy.linalg import svd

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

# Crear una sola figura para mostrar las imágenes reconstruidas
plt.figure(figsize=(16, 8))

for i, image_file_name in enumerate(image_file_names):
    try:
        # Cargar la imagen
        image = mpimg.imread(image_file_name)

        # Aplicar SVD a la matriz de la imagen
        U, S, VT = np.linalg.svd(image, full_matrices=False)

        # Reconstruir la imagen original utilizando las primeras dimensiones
        reconstructed_matrix_primero = np.dot(U[:, :n_componentes_primero], np.dot(np.diag(S[:n_componentes_primero]), VT[:n_componentes_primero, :]))

        # Reconstruir la imagen original utilizando las últimas dimensiones
        reconstructed_matrix_ultimo = np.dot(U[:, -n_componentes_ultimo:], np.dot(np.diag(S[-n_componentes_ultimo:]), VT[-n_componentes_ultimo:, :]))

        # Crear subparcelas para las imágenes
        plt.subplot(len(image_file_names), 4, i * 4 + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Imagen Original', fontsize=10)
        plt.axis('off')

        plt.subplot(len(image_file_names), 4, i * 4 + 2)
        plt.imshow(reconstructed_matrix_primero, cmap='gray')
        plt.title(f'Primeras {n_componentes_primero} dimensiones', fontsize=10)
        plt.axis('off')

        plt.subplot(len(image_file_names), 4, i * 4 + 3)
        plt.imshow(reconstructed_matrix_ultimo, cmap='gray')
        plt.title(f'Últimas {n_componentes_ultimo} dimensiones', fontsize=10)
        plt.axis('off')

    except Exception as e:
        print(f"Error al procesar la imagen: {image_file_name}. Detalles del error: {e}")

plt.tight_layout()
plt.show()
