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
    const img = new Image();
    img.src = imageData;
    await img.decode();

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 150;
    canvas.height = 150;
    ctx.drawImage(img, 0, 0, 150, 150);
    const imageDataArray = ctx.getImageData(0, 0, 150, 150).data;

    // Convert to a float32 array
    const float32Array = new Float32Array(150 * 150 * 3);
    for (let i = 0; i < 150 * 150; i++) {
        for (let j = 0; j < 3; j++) {
            float32Array[i * 3 + j] = imageDataArray[i * 4 + j] / 255;
        }
    }

    // Create the ONNX Tensor
    const tensor = new ort.Tensor('float32', float32Array, [1, 3, 150, 150]);
    return tensor;
}
