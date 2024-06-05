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
                const inputTensor = await preprocessImageData(imageData);

                // Load ONNX model and run inference
                try {
                    const session = await ort.InferenceSession.create('model/model.onnx');
                    const inputName = session.inputNames[0]; // Get the input name dynamically
                    const feeds = {};
                    feeds[inputName] = inputTensor;

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
            const input = new Float32Array(150 * 150 * 3);
            for (let i = 0; i < 150 * 150; i++) {
                input[i * 3] = imageData.data[i * 4] / 255;
                input[i * 3 + 1] = imageData.data[i * 4 + 1] / 255;
                input[i * 3 + 2] = imageData.data[i * 4 + 2] / 255;
            }

            // Convert to Tensor
            const tensorInput = new ort.Tensor('float32', input, [1, 150, 150, 3]);
            resolve(tensorInput);
        };

        img.onerror = (error) => {
            reject(error);
        };

        img.src = imageData;
    });
}
