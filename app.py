from flask import Flask, jsonify, request
from flask_cors import CORS
from newspaper import Article
import joblib
import re
import logging

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model.pkl')

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'url' not in data:
            return jsonify({'error': 'URL diperlukan.'}), 400
        
        url = data['url']
        logging.debug(f"URL received: {url}")

        # Attempt to download and process the article
        try:
            article = Article(url)
            article.download()
            article.parse()
            article_text = article.text
            if not article_text:
                logging.debug(f"Article HTML: {article.html[:1000]}...")  # log first 1000 chars of HTML for debugging
                return jsonify({'error': 'Gagal mengekstrak teks dari artikel. Mohon cek URL anda dan coba lagi.'}), 400
            logging.debug(f"Article text: {article_text[:500]}...")  # log first 500 chars
        except Exception as e:
            logging.error(f"Failed to process article: {str(e)}")
            return jsonify({'error': f'Gagal memproses artikel: {str(e)}'}), 400

        # Preprocess the text
        try:
            article_text = re.sub(r'\W', ' ', article_text)
            article_text = re.sub(r'\s+', ' ', article_text)
            article_text = article_text.lower()
            logging.debug(f"Processed text: {article_text[:500]}...")  # log first 500 chars
        except Exception as e:
            logging.error(f"Gagal memproses ulang artikel: {str(e)}")
            return jsonify({'error': f'Gagal memproses ulang artikel: {str(e)}'}), 500

        # Predict
        try:
            prediction = model.predict([article_text])
            result = 'Asli' if prediction[0] == 1 else 'Hoaks'
            logging.debug(f"Hasil prediksi: {result}")
        except Exception as e:
            logging.error(f"Prediksi gagal: {str(e)}")
            return jsonify({'error': f'Gagal memprediksi: {str(e)}'}), 500
        
        return jsonify({'prediction': result})
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
