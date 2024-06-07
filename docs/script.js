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

                // Preprocess the image data
                const inputTensor = await preprocessImageData(imageData);

                // Load TensorFlow.js model and run inference
                try {
                    const model = await tf.loadLayersModel('model/model.json');
                    const predictions = model.predict(inputTensor);
                    const output = predictions.dataSync()[0];

                    // Display the result
                    const classification = output > 0.5 ? 'Healthy' : 'Bleached';
                    const probability = output;
                    sessionStorage.setItem('prediction', `Output: ${classification}, Probability Score: ${probability}`);
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
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.src = imageData;
        img.onload = () => {
            const canvas = document.createElement('canvas');
            canvas.width = 150;
            canvas.height = 150;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, 150, 150);
            const imageData = ctx.getImageData(0, 0, 150, 150);

            const data = tf.browser.fromPixels(imageData).div(255.0);
            const resized = tf.image.resizeBilinear(data, [150, 150]);
            const expanded = resized.expandDims(0);
            resolve(expanded);
        };
        img.onerror = (err) => {
            reject(err);
        };
    });
}
