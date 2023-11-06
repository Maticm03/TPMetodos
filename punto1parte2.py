import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Lista de rutas de archivos de imagen
image_file_names = [
    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img00.jpeg",
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
    "C:/Users/usuario/OneDrive/Escritorio/Métodos numéricos y optimización/TP3/img15.jpeg"
]

# Valor máximo permitido para el error (10%)
max_error = 0.10

# Cargar una imagen para calcular la transformación
sample_image = mpimg.imread(image_file_names[0])

# Realizar SVD en la imagen de ejemplo
U, S, VT = np.linalg.svd(sample_image, full_matrices=False)

# Encontrar el valor mínimo de "d" que cumple con el límite de error para la transformación
d = 1
error = 1.0

while error > max_error:
    compressed_matrix = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    error = np.linalg.norm(sample_image - compressed_matrix, 'fro') / np.linalg.norm(sample_image, 'fro')
    d += 1

# Calcular el error para todas las imágenes con el mismo valor "d"
errors = []

for image_file_name in image_file_names:
    image = mpimg.imread(image_file_name)
    compressed_matrix = np.dot(U[:, :d], np.dot(np.diag(S[:d]), VT[:d, :]))
    error = np.linalg.norm(image - compressed_matrix, 'fro') / np.linalg.norm(image, 'fro')
    errors.append(error)

# Mostrar el valor mínimo de "d"
print(f"El número mínimo de dimensiones 'd' para cumplir con el límite de error es: {d - 1}")

# Graficar el error para todas las imágenes
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(image_file_names) + 1), errors, marker='o', linestyle='-', color='b')
plt.title('Error de Compresión vs. Imagen')
plt.xlabel('Imagen')
plt.ylabel('Error de Compresión')
plt.xticks(range(1, len(image_file_names) + 1))
plt.grid(True)
plt.show()
