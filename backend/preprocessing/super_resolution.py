import cv2


def super_resolve_image(input_path, output_path):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Could not read enhanced image")

    height, width = image.shape

    super_resolved = cv2.resize(
        image,
        (width * 2, height * 2),
        interpolation=cv2.INTER_LANCZOS4
    )

    cv2.imwrite(output_path, super_resolved)

    return output_path