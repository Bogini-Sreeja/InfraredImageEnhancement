import cv2
import os


def preprocess_image(input_path, output_path):

    # Read image
    image = cv2.imread(input_path)

    if image is None:
        raise ValueError("Unable to read image")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Resize image
    resized = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # 3. Normalize image
    normalized = cv2.normalize(
        resized,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    # 4. Histogram equalization
    equalized = cv2.equalizeHist(normalized)

    # Save result
    cv2.imwrite(
        output_path,
        equalized
    )

    return output_path