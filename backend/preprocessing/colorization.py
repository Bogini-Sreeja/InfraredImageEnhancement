import cv2
import os


def colorize_image(input_path, output_path):

    # Read the super-resolved image
    image = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {input_path}"
        )

    # Apply a thermal-style color mapping
    colorized = cv2.applyColorMap(
        image,
        cv2.COLORMAP_TURBO
    )

    # Create output directory
    directory = os.path.dirname(output_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    # Save RGB-style colorized image
    success = cv2.imwrite(
        output_path,
        colorized
    )

    if not success:
        raise ValueError(
            "Could not save colorized image."
        )

    return output_path