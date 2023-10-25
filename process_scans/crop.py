import cv2
import numpy as np
import os

def split_and_center_characters(image_path, output_folder):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Precrop the image to get rid of UI stuff around the outside
    centre_x = image.shape[1] // 2
    centre_y = image.shape[0] // 2
    target_width = 12304
    target_height = 12256
    image = image[centre_y - (target_height+1)//2 : centre_y + target_height//2, centre_x - (target_width+1)//2 : centre_x + target_width//2]

    # Get the height and width of the input image
    height, width = image.shape

    # Define the size of each grid cell
    cell_height = height // 5
    cell_width = width // 5

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through the 5x5 grid and process each character
    count = 0
    for i in range(5):
        for j in range(5):
            # Calculate the position of the current grid cell
            x = j * cell_width
            y = i * cell_height

            # Crop the character from the input image
            cropped = image[y:y + cell_height, x:x + cell_width]

            # Invert the cropped image (white to black, black to white)
            inverted = cv2.bitwise_not(cropped)

            # Threshold the inverted image
            _, thresh = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get the bounding rectangle of the largest contour (the character)
            largest_cnt = max(contours, key=cv2.contourArea)
            x_char, y_char, w_char, h_char = cv2.boundingRect(largest_cnt)

            # Crop the character from the inverted image
            cropped_char = inverted[y_char:y_char + h_char, x_char:x_char + w_char]

            # Create a centered blank image with padding (white background)
            max_dim = max(cell_width, cell_height)
            centered = np.full((max_dim, max_dim), 255, dtype=np.uint8)

            # Calculate the position of the character in the centered image
            x_offset = (max_dim - w_char) // 2
            y_offset = (max_dim - h_char) // 2

            # Place the cropped character in the centered image
            centered[y_offset:y_offset + h_char, x_offset:x_offset + w_char] = cropped_char

            # Save the centered character
            count += 1
            output_file = os.path.join(output_folder, f"{count}.png")
            cv2.imwrite(output_file, centered)

            print(f"Processed: {output_file}")

def center_black_pixels_in_images(input_folder):
    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file in the input folder
    for filename in files:
        # Check if the filename represents an integer number
        try:
            int(filename.split(".")[0])
        except ValueError:
            continue

        # Get the file path
        file_path = os.path.join(input_folder, filename)

        # Read the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Get the height and width of the image
        height, width = image.shape

        # Threshold the image
        _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

        # Find the bounding rectangle of all black pixels (the character)
        x_char, y_char, w_char, h_char = cv2.boundingRect(thresh)

        # Crop the character from the image
        cropped_char = image[y_char:y_char + h_char, x_char:x_char + w_char]

        # Create a centered blank image with padding (white background)
        centered = np.full((height, width), 255, dtype=np.uint8)

        # Calculate the position of the character in the centered image
        x_offset = (width - w_char) // 2
        y_offset = (height - h_char) // 2

        # Place the cropped character in the centered image
        centered[y_offset:y_offset + h_char, x_offset:x_offset + w_char] = cropped_char

        # Save the centered character, overwriting the original image
        cv2.imwrite(file_path, centered)

        print(f"Processed: {file_path}")

def crop_images(input_folder):
    # List all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file in the input folder
    for filename in files:
        # Check if the filename represents an integer number
        try:
            int(filename.split(".")[0])
        except ValueError:
            continue

        # Get the file path
        file_path = os.path.join(input_folder, filename)

        # Read the image
        image = cv2.imread(file_path)

        # Get the image dimensions
        height, width, _ = image.shape

        # Calculate the crop boundaries
        x_start = 460
        y_start = 460
        x_end = width - 460
        y_end = height - 460

        # Crop the image
        cropped_image = image[y_start:y_end, x_start:x_end]

        # Save the cropped image back to the same file
        cv2.imwrite(file_path, cropped_image)

        print(f"Cropped: {file_path}")

# Specify the input image path
img_folder = r"E:\greg\1st batch\ct images"

for idx, file in enumerate(os.scandir(img_folder)):
    # Specify the output folder path
    output_folder = fr"E:\greg\1st batch\myruns\r{idx}"

    # Execute the first code
    split_and_center_characters(os.path.join(img_folder, file.name), output_folder)

    # Execute the second code
    input_folder = output_folder
    center_black_pixels_in_images(input_folder)

    # Execute the third code
    # crop_images(input_folder)
