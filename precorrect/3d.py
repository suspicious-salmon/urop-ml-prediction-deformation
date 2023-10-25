"""Converts black-and-white PNG images (black content, white everything else) to STL files at a scale of 1mm per pixel, with depth extrusion_length mm."""

#%%
import os
import numpy as np
from PIL import Image
import cv2
from stl import mesh
#%%
def load_grayscale_image(image_path):
    return Image.open(image_path).convert("L")

def create_binary_mask(image, threshold=128):
    _, binary_mask = cv2.threshold(np.array(image), threshold, 255, cv2.THRESH_BINARY)
    return binary_mask

def create_mesh(binary_mask, pixel_size=1, extrusion_length=10):
    vertices = []
    faces = []

    for i in range(binary_mask.shape[0] - 1):
        for j in range(binary_mask.shape[1] - 1):
            if binary_mask[i][j] == 0:
                x = j * pixel_size
                y = (binary_mask.shape[0] - i) * pixel_size  # Reverse the y-coordinate
                z = 0

                idx = len(vertices)

                vertices.extend([
                    (x, y, z),
                    (x + pixel_size, y, z),
                    (x + pixel_size, y - pixel_size, z),  # Reverse the y-coordinate
                    (x, y - pixel_size, z),  # Reverse the y-coordinate
                    (x, y, z + extrusion_length),
                    (x + pixel_size, y, z + extrusion_length),
                    (x + pixel_size, y - pixel_size, z + extrusion_length),  # Reverse the y-coordinate
                    (x, y - pixel_size, z + extrusion_length)  # Reverse the y-coordinate
                ])

                faces.extend([
                    (idx, idx + 1, idx + 2), (idx, idx + 2, idx + 3),
                    (idx + 4, idx + 5, idx + 6), (idx + 4, idx + 6, idx + 7),
                    (idx, idx + 1, idx + 5), (idx, idx + 5, idx + 4),
                    (idx + 1, idx + 2, idx + 6), (idx + 1, idx + 6, idx + 5),
                    (idx + 2, idx + 3, idx + 7), (idx + 2, idx + 7, idx + 6),
                    (idx + 3, idx, idx + 4), (idx + 3, idx + 4, idx + 7)
                ])

    vertices_np = np.array(vertices, dtype=np.float32)
    faces_np = np.array(faces, dtype=np.uint32)

    mesh_data = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices_np[f[j], :]

    return mesh_data


def save_mesh_to_stl(mesh_data, output_file):
    mesh_data.save(output_file)

def png_to_stl(image_path, output_file, extrusion_length=10):
    grayscale_image = load_grayscale_image(image_path)
    binary_mask = create_binary_mask(grayscale_image)
    mesh_data = create_mesh(binary_mask, extrusion_length=extrusion_length)
    save_mesh_to_stl(mesh_data, output_file)

def batch_convert_png_to_stl(png_dir, stl_dir, extrusion_length=10):
    if not os.path.exists(stl_dir):
        os.makedirs(stl_dir)

    for file in os.listdir(png_dir):
        if file.endswith(".png") or file.endswith(".tif"):
            input_image_path = os.path.join(png_dir, file)
            output_stl_path = os.path.join(stl_dir, f"{os.path.splitext(file)[0]}.stl")
            png_to_stl(input_image_path, output_stl_path, extrusion_length)

if __name__ == "__main__":
    input_png_directory = r"E:\greg\Chinese Characters\3D Printed Deformations\urop-structured-nn\other\Downscaled"
    output_stl_directory = r"E:\greg\Chinese Characters\3D Printed Deformations\urop-structured-nn\other\Pre-Corrected_STLs"
    extrusion_length = 10
    batch_convert_png_to_stl(input_png_directory, output_stl_directory, extrusion_length)

# %%
