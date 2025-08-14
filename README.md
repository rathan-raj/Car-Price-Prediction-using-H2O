Car Price Prediction using H2O AutoML
This project predicts used car prices in USD using H2O.ai's AutoML framework. It leverages machine learning to train, evaluate, and serve predictive models, enabling accurate and efficient car price estimations.
🚀 Features
- H2O AutoML for automated model selection and tuning.
- Predicts prices based on car attributes (make, model, year, mileage, etc.).
- Ready-to-use Python API for predictions.
- Easy to run locally or deploy to cloud platforms.
📂 Project Structure

app.py                  # Flask/Gradio app for serving predictions
AML3304_Project_.ipynb  # Jupyter Notebook for training & evaluation
requirements.txt        # Project dependencies
README.md               # Project documentation
model/                  # Saved H2O model files

⚙️ Installation
1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
🧠 Model Training
The training pipeline is implemented in the Jupyter notebook: AML3304_Project_.ipynb

Steps:
1. Load dataset (CSV or database).
2. Preprocess & clean data.
3. Use H2OAutoML to train multiple models.
4. Evaluate using RMSE, MAE, and R² score.
5. Save the best-performing model using:
h2o.save_model(model=best_model, path='./model', force=True)
📊 Example Prediction
import h2o
model = h2o.load_model('model/GBM_grid_1_AutoML_...')

data = h2o.H2OFrame({
    'year': [2015],
    'mileage': [45000],
    'make': ['Toyota'],
    'model': ['Camry'],
    'state': ['CA'],
    'condition': ['good']
})

pred = model.predict(data)
print(pred)
▶️ Running the App
Run the API/Gradio app:
python app.py
This will start a local server (default: http://127.0.0.1:5000) or Gradio interface for predictions.
📦 Deployment
You can deploy this project to:
- Hugging Face Spaces
- Streamlit Cloud
- Heroku
- Docker

Example: Hugging Face Deployment
gradio deploy --space <your-hf-username>/<space-name>
📜 Requirements
Python 3.8+
h2o
pandas
numpy
scikit-learn
gradio or flask (for serving)
Jupyter (for training)

Install all with:
pip install -r requirements.txt
🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
✨ Acknowledgments
H2O.ai AutoML - https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
Gradio - https://gradio.app/
Dataset source: (Add your dataset link here)
<img width="432" height="643" alt="image" src="https://github.com/user-attachments/assets/5f2bf731-68bc-4ca7-818a-27a7d2c9f629" />
