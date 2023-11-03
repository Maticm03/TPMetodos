import numpy as np
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

# Cargar la imagen original (por ejemplo, la primera imagen)
image_file_name = "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img00.jpeg"
original_image = mpimg.imread(image_file_name)

# Valor máximo permitido para el error (10%)
max_error = 0.10

# Inicializa d a 1
d = 1

# Itera para encontrar el mínimo d
while True:
    try:
        # Aplicar SVD a la matriz de la imagen con d dimensiones
        U, S, VT = np.linalg.svd(original_image, full_matrices=False)
        compressed_matrix = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))

        # Calcular el error entre la imagen original y la imagen reconstruida bajo la norma de Frobenius
        error = np.linalg.norm(original_image - compressed_matrix, 'fro') / np.linalg.norm(original_image, 'fro')

        # Comprobar si el error es menor o igual al 10%
        if error <= max_error:
            break

        # Incrementar d para probar con una dimensión adicional
        d += 1

    except Exception as e:
        print(f"Error al procesar la imagen con d={d}. Detalles del error: {e}")

print(f"El valor mínimo de d para la primera imagen es: {d}")
print(f"Error para la primera imagen con d={d}: {error}")

# Ahora aplica la misma compresión a todas las imágenes del conjunto
compressed_images = []
for image_file_name in image_file_names:
    try:
        image = mpimg.imread(image_file_name)
        U, S, VT = np.linalg.svd(image, full_matrices=False)
        compressed_matrix = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
        compressed_images.append(compressed_matrix)
    except Exception as e:
        print(f"Error al procesar la imagen: {image_file_name}. Detalles del error: {e}")
        
# Calcula y muestra el error para el resto de las imágenes del conjunto
for i, image_file_name in enumerate(image_file_names[1:], start=1):
    try:
        image = mpimg.imread(image_file_name)
        U, S, VT = np.linalg.svd(image, full_matrices=False)
        compressed_matrix = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
        error = np.linalg.norm(image - compressed_matrix, 'fro') / np.linalg.norm(image, 'fro')
        print(f"Error para la imagen {i} con d={d}: {error}")
    except Exception as e:
        print(f"Error al procesar la imagen {i}: {image_file_name}. Detalles del error: {e}")
