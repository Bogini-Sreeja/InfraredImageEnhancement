import cv2
import numpy as np


def preprocess_image(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not read input image")

    image = np.nan_to_num(image, nan=0.0)

    image = cv2.normalize(
        image,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    cv2.imwrite(output_path, image)

    return output_path