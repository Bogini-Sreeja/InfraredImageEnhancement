function ImagePreview({ image }) {
  if (!image) {
    return null;
  }

  const imageURL = URL.createObjectURL(image);

  return (
    <div className="preview-container">
      <h2>Original IR Image</h2>

      <img
        src={imageURL}
        alt="Uploaded infrared satellite image"
        className="preview-image"
      />

      <p className="image-name">
        File: {image.name}
      </p>

      <p className="image-size">
        Size: {(image.size / 1024).toFixed(2)} KB
      </p>
    </div>
  );
}

export default ImagePreview;