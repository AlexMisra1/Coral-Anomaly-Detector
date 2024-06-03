document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('imageInput');
    const errorDiv = document.getElementById('error');
    const predictionParagraph = document.getElementById('prediction');

    if (uploadForm) {
        uploadForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            if (imageInput.files.length === 0) {
                errorDiv.textContent = 'Please upload an image.';
                return;
            }

            const file = imageInput.files[0];
            const reader = new FileReader();
            reader.onload = async () => {
                const imageData = reader.result;

                // Preprocess the image data here (resize, normalize, etc.)
                const inputTensor = preprocessImageData(imageData);

                // Load ONNX model and run inference
                try {
                    const session = new onnx.InferenceSession();
                    await session.loadModel('docs/model/model.onnx');
                    const feeds = { input: inputTensor };
                    const results = await session.run(feeds);
                    const outputTensor = results.values().next().value;

                    // Display the result
                    sessionStorage.setItem('prediction', `Output: ${outputTensor.data}`);
                    window.location.href = 'result.html';
                } catch (err) {
                    errorDiv.textContent = 'Error running model: ' + err.message;
                }
            };
            reader.readAsDataURL(file);
        });
    }

    if (predictionParagraph) {
        const prediction = sessionStorage.getItem('prediction');
        predictionParagraph.textContent = prediction;
    }
});

async function preprocessImageData(imageData) {
    // Preprocess the image data (e.g., resize, normalize)
    // You can use TensorFlow.js or any other library for this step
    // Example: convert imageData to a tensor and resize it to match the input size of the model

    // Assuming imageData is already preprocessed as needed

    // Load the ONNX model
    const model = await loadONNXModel('model/model.onnx');

    // Convert the preprocessed image data to a tensor
    const tensor = preprocessImageDataToTensor(imageData);

    // Run inference using the ONNX model
    const output = await model.predict(tensor);

    // Postprocess the output (e.g., get predictions)
    // Example: get the predicted class from the output tensor

    // Assuming output is postprocessed as needed

    return output;
}

async function loadONNXModel(modelPath) {
    // Load the ONNX model
    const model = await onnx.load(modelPath);

    // Create a session from the model
    const session = await model.createSession();

    // Define a predict function using the session
    const predict = async (inputData) => {
        // Run inference using the session
        const outputData = await session.run(inputData);

        // Return the output data
        return outputData;
    };

    // Return the predict function
    return predict;
}

function preprocessImageDataToTensor(imageData) {
    // Convert imageData to a tensor and preprocess as needed
    // Example: convert imageData to a tensor using TensorFlow.js
    // Ensure the tensor matches the input shape and data format expected by the model

    // Assuming imageData is already converted to a tensor

    return tensor;
}




