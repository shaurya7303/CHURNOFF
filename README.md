CHURN OFF – Customer Churn Prediction App
CHURN OFF is an interactive Streamlit web application that predicts whether a bank customer is likely to churn based on their profile and account features. The app uses a trained TensorFlow model and standard preprocessing artifacts to generate real-time churn risk scores.​

Features
📊 Real-time customer churn prediction using a trained TensorFlow model (churn_model.h5).​

🎛️ Interactive UI with sliders, dropdowns, and numeric inputs for customer attributes.

📈 Dynamic probability display with risk status (high/low churn risk).

🧮 Built-in preprocessing using saved encoders and scaler (le_gender.pkl, ohe_geography.pkl, scaler.pkl).

Tech Stack
Python 3.x

TensorFlow / Keras

scikit-learn

pandas, numpy

Streamlit

Project Structure
text
.
├── app.py                # Streamlit application (CHURN OFF UI)
├── churn_model.h5        # Trained Keras/TensorFlow churn model
├── le_gender.pkl         # LabelEncoder for Gender
├── ohe_geography.pkl     # OneHotEncoder for Geography
├── scaler.pkl            # StandardScaler for numerical + encoded features
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
Getting Started
1. Clone the repository
bash
git clone <https://github.com/shaurya7303/CHURNOFF>.git
cd <Churn Predictor > # or whatever u want to name the folder
2. Create a virtual environment (recommended)
bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# or
.\.venv\Scripts\activate         # Windows
3. Install dependencies
Make sure requirements.txt contains at least:

text
streamlit
tensorflow
pandas
numpy
scikit-learn
Then install:

bash
pip install -r requirements.txt
4. Place model & artifacts
Ensure the following files are present in the project root, trained on the same feature pipeline used in the app:

churn_model.h5

le_gender.pkl

ohe_geography.pkl

scaler.pkl

These are required for the app to load and predict correctly.​

5. Run the app
bash
streamlit run app.py
By default, Streamlit will launch at:

Local: http://localhost:8501

How It Works
User selects or enters customer attributes (geography, gender, age, tenure, balance, products, credit card, active member status, salary, credit score).

The app uses the saved encoders and scaler to transform the input into the same feature space used during training.

The TensorFlow model outputs a churn probability between 0 and 1.

The UI displays:

Churn probability

Stay probability

A textual label (e.g., “High Churn Risk” or “Low Churn Risk”) with color-coded status

Deployment
You can deploy this app using platforms that support Streamlit, such as:

Streamlit Community Cloud

Docker + any cloud provider

VM / server with Python and streamlit installed

For Streamlit Community Cloud, push this project to GitHub and configure:

Repository: your-repo

Main file: app.py

Python version & dependencies: from requirements.txt​

Future Improvements
Add model explainability (e.g., SHAP) to show feature importance per prediction.​

Log predictions for monitoring and feedback.

Support batch predictions from CSV upload.