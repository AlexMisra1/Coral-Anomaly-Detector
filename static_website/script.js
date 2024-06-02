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
                    await session.loadModel('model/model.onnx');
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

function preprocessImageData(imageData) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    return new Promise((resolve, reject) => {
        img.onload = () => {
            canvas.width = 150;
            canvas.height = 150;
            ctx.drawImage(img, 0, 0, 150, 150);
            const imageData = ctx.getImageData(0, 0, 150, 150);

            // Normalize pixel values to range [0, 1]
            const input = new Float32Array(imageData.data.length / 4);
            for (let i = 0; i < input.length; i++) {
                input[i] = imageData.data[i * 4] / 255;  // Assuming image is in grayscale
            }

            // Reshape and convert to Tensor
            const tensorInput = new onnx.Tensor(input, 'float32', [1, 150, 150, 3]);
            resolve(tensorInput);
        };

        img.onerror = (error) => {
            reject(error);
        };

        img.src = imageData;
    });
}



