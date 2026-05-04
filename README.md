# 🛫 DelayX: Flight Delay Prediction & Recommendation System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![HTML/CSS](https://img.shields.io/badge/HTML5_&_CSS3-E34F26?style=for-the-badge&logo=html5&logoColor=white)

DelayX is a full-stack Machine Learning web application designed to predict flight delays and recommend optimal travel options. Built with a robust Python backend and an intuitive front-end interface, DelayX helps travelers make informed decisions to avoid disruptions and save time.

---

## ✨ Key Features
- **🔮 Real-Time Delay Prediction:** Input flight details to instantly get the probability of a delay.
- **🚦 Risk Classification:** Flights are categorized into intuitively color-coded risk levels (Low, Medium, High Risk) for quick decision-making.
- **✈️ Smart Recommendation System:** If your selected flight has a high chance of delay, DelayX automatically suggests alternative airlines or routes with better on-time performance.
- **📊 Historical Context:** Provides average delay times and potential causes based on historical data.

---

## 🛠️ Tech Stack
- **Backend Framework:** Python, Flask
- **Machine Learning:** Scikit-learn
- **Data Manipulation:** Pandas, NumPy
- **Frontend UI:** HTML, Vanilla CSS (Responsive & Dynamic Design)

---

## 🧠 Machine Learning Models Used
DelayX evaluates multiple machine learning models to ensure optimal accuracy in predicting flight delays:
- **Logistic Regression:** For baseline probabilistic classification.
- **K-Nearest Neighbors (KNN):** To identify patterns based on similar historical flights.
- **Support Vector Machines (SVM):** For complex boundary decision-making.
- **Random Forest:** An ensemble method used to achieve maximum predictive accuracy.

---

## 📈 Model Performance
After extensive training and testing across multiple models, **Random Forest** proved to be the most effective for this dataset, achieving an impressive **~92% accuracy rate**. This model is integrated into the live web application to power all predictions.

---

## 📂 Dataset Description
The model is trained on a comprehensive dataset of millions of historical commercial flights. Key features used for training include:
- `AIRLINE_CODE`: Operating airline carrier
- `ORIGIN` / `DEST`: Departure and arrival airports
- `MONTH` / `DAY_OF_WEEK`: Seasonal and weekly travel patterns
- `DEP_DELAY`: Historical departure delays
- `DEP_HOUR` / `IS_WEEKEND`: Time-based features affecting air traffic

*(Note: The full dataset files are extremely large and are not included in this repository to comply with size limits.)*

---

## 🏗️ System Architecture
1. **User Input:** The user submits flight details (Origin, Destination, Airline, Date/Time) via the web interface.
2. **Backend Processing:** Flask processes the request and formats the data for the model.
3. **ML Prediction:** The optimized Random Forest model evaluates the input and generates a delay probability.
4. **Recommendation Engine:** If the delay risk is high, alternative flights are queried and evaluated.
5. **Output:** The UI dynamically renders the verdict, probabilities, and recommendations back to the user.

---

## ⚙️ Installation Steps

Follow these steps to run the DelayX system on your local machine.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/OmGupta723/DelayX.git
   cd DelayX
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   *(Ensure you have a `requirements.txt` or install the necessary packages manually)*
   ```bash
   pip install flask pandas scikit-learn numpy
   ```

4. **Run the Flask Application:**
   ```bash
   cd website
   python app.py
   ```

5. The application will be accessible at `http://127.0.0.1:5000`.

---

## 🚀 Usage Instructions
1. Navigate to the homepage.
2. Enter your travel details, including the airline, origin, destination, and expected travel date.
3. Click **"Predict Delay"**.
4. View your flight's delay probability, risk classification, and potential causes.
5. If the flight is deemed high-risk, browse the recommended alternatives to find a better flight!

---

## 📸 Screenshots
> *Placeholder for UI Screenshots. Add images of the homepage, prediction results, and recommendation page here!*

![Homepage Placeholder](https://via.placeholder.com/800x400?text=DelayX+Homepage)
![Prediction Result Placeholder](https://via.placeholder.com/800x400?text=DelayX+Prediction+Result)

---

## 🔮 Future Improvements
- **Live Data Integration:** Connect to live aviation APIs (e.g., FlightAware, AviationStack) for real-time weather and air traffic data.
- **Deep Learning Upgrade:** Implement Neural Networks (TensorFlow/PyTorch) to capture more complex non-linear patterns in global flight data.
- **Cloud Deployment:** Deploy the application and model to AWS, GCP, or Heroku for global accessibility.
- **User Accounts:** Expand the profile section to allow users to save upcoming trips and receive automated delay alerts.

---

## 👨‍💻 Author
**Om Gupta**
- GitHub: [@OmGupta723](https://github.com/OmGupta723)

*If you like this project, please consider giving it a ⭐!*
