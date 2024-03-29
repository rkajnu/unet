from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import numpy as np

# Assuming the `Infer` class is defined in a separate module
from predictor import Infer

app = Flask(__name__)

# Initialize the Infer class with the checkpoint path
chkpt_path = "unet_membrane.keras"
infer = Infer(chkpt_path)


@app.route("/infer", methods=["POST"])
def infer_image():
    # Get the image from the request
    file = request.files["image"]
    img = Image.open(file.stream)

    # Convert the image to a numpy array
    input_data = np.array(img)

    # Perform inference
    thresh = 0.5  # Set your threshold here
    mask = infer.predictor(input_data, thresh)

    # Create an image from the mask
    result_img = Image.fromarray(mask.astype("uint8"))

    # Convert the result image to bytes
    img_byte_array = BytesIO()
    result_img.save(img_byte_array, format="PNG")
    img_byte_array = img_byte_array.getvalue()

    return jsonify({"result_image": img_byte_array.decode("latin-1")})


if __name__ == "__main__":
    app.run(debug=True)
