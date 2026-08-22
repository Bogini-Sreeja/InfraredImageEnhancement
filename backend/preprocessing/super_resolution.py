import cv2
import os


def super_resolve_image(input_path, output_path):

    image = cv2.imread(
        input_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {input_path}"
        )

    height, width = image.shape[:2]

    new_width = width * 2
    new_height = height * 2

    super_resolved = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4
    )

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        super_resolved
    )

    if not success:
        raise ValueError(
            "Could not save super-resolution image."
        )

    return output_path