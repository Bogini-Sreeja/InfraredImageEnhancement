function ImageUploader({ onImageUpload }) {
  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    // Check whether the selected file is an image
    if (!file.type.startsWith("image/")) {
      alert("Please select a valid image file.");
      return;
    }

    onImageUpload(file);
  };

  return (
    <div className="upload-container">
      <label htmlFor="image-upload" className="upload-button">
        Upload IR Image
      </label>

      <input
        id="image-upload"
        type="file"
        accept="image/*"
        onChange={handleImageChange}
        hidden
      />
    </div>
  );
}

export default ImageUploader;