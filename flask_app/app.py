from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import torch
import numpy as np
import cv2
device_in_use = 'cpu'

# Default is homography model
# Change def submit() is needed to run rotation model through flask
m1 = torch.load("homography.pth", map_location=torch.device('cpu'))

def normalize(tensor):
    tensor = tensor.clone()  # Create a copy of the tensor
    mean = 0.1307
    std = 0.3081

    tensor.sub_(mean).div_(std)  # Subtract mean and divide by std for the whole tensor
    return tensor

def tensor_to_cv(tensor):
    # Ensure tensor is detached and on CPU
    numpy_image = tensor.detach().cpu().numpy()
    
    # Handle grayscale and color images differently
    if tensor.shape[0] == 1:  # Grayscale image, single channel
        # Remove channel dimension for grayscale, resulting in HxW
        numpy_image = numpy_image.squeeze()
    else:  # Color image, 3 channels
        # Convert from PyTorch's CHW format to OpenCV's HWC format
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
    
    # Convert from [0, 1] to [0, 255] and change to uint8
    cv_image = (numpy_image * 255).astype(np.uint8)
    return cv_image 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
  
@app.route('/submit', methods=['POST'])
def submit():
    # Extract the base64 image data from the request
    image_data = request.json['image_data'] 
    # Remove the header from the base64 string
    image_data = image_data.split(",")[1]
    # Decode the base64 string
    image_bytes = base64.b64decode(image_data)
    # Convert bytes to a PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # Convert the image to grayscale
    image = image.convert('L')

    # Resize the image to 28x28 pixels
    image = image.resize((28, 28), Image.LANCZOS)


    # Convert the PIL Image to a PyTorch Tensor
    tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
    tensor = tensor.unsqueeze(0)

    # Normalize the tensor
    tensor_normalized = normalize(tensor)

    # Add a batch dimension
    tensor_normalized = tensor_normalized.unsqueeze(0)


    norm_transformed_image_batched = tensor_normalized.to(device_in_use)

    with torch.no_grad():
        outputs = m1(norm_transformed_image_batched).cpu()

    transformed_image_cv = tensor_to_cv(tensor)

    outputs = outputs.cpu()

    # Assuming 'outputs' contains the 8 parameters of the transformation matrix
    predicted_params = outputs[0].numpy()  # Convert to numpy array if it's not already
    predicted_inverse_matrix = predicted_params.reshape((3, 3))  # Append 1 for the last element and reshape


    # Apply the predicted inverse transformation to the transformed image
    # Note: 'cv_image' is obtained from 'tensor_to_cv(transformed_image.cpu())', as in your code
    pred_restored = cv2.warpPerspective(transformed_image_cv, predicted_inverse_matrix, (transformed_image_cv.shape[0], transformed_image_cv.shape[1]))

    # Convert the processed image back to base64
    processed_image = Image.fromarray(pred_restored)
    buffered = io.BytesIO()
    processed_image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({'image_data': f'data:image/png;base64,{encoded_image}'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)

