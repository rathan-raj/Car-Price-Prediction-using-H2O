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
git clone https://github.com/<your-username>/<your-repo>.git](https://github.com/rathan-raj/Car-Price-Prediction-using-H2O)

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

📦 Deployment
You can deploy this project to:
- Hugging Face Spaces
- Streamlit Cloud
- Heroku
- Docker

Install all with:
pip install -r requirements.txt
🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
