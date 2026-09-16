import cv2


def enhance_image(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not read preprocessed image")

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    cv2.imwrite(output_path, enhanced)

    return output_path