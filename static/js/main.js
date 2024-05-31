document.getElementById('url-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const url = document.getElementById('url-input').value;
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: url })
        });
  
        const resultDiv = document.getElementById('result');
  
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network response was not ok');
        }
  
        const result = await response.json();
  
        if (result.error) {
            resultDiv.className = 'alert alert-danger';
            resultDiv.innerText = `Error: ${result.error}`;
        } else {
            resultDiv.className = result.prediction === 'Asli' ? 'alert alert-success' : 'alert alert-danger';
            resultDiv.innerText = `Prediksi: ${result.prediction}`;
        }
        resultDiv.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        const resultDiv = document.getElementById('result');
        resultDiv.className = 'alert alert-danger';
        resultDiv.innerText = `An error occurred: ${error.message}`;
        resultDiv.style.display = 'block';
    }
  });
  