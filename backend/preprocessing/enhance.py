import cv2
import os


def enhance_image(input_path, output_path):

    # Read image
    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {input_path}"
        )

    # Improve contrast using CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    # Slight sharpening
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    enhanced = cv2.detailEnhance(
        cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        ),
        sigma_s=10,
        sigma_r=0.15
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_BGR2GRAY
    )

    # Create output directory
    directory = os.path.dirname(
        output_path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    # Save
    success = cv2.imwrite(
        output_path,
        enhanced
    )

    if not success:
        raise ValueError(
            "Could not save enhanced image."
        )

    return output_path