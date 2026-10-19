<div align="center">

<h1>🔥 CalorIQ</h1>
<h3><em>Intelligent Calorie Burn Prediction Using Machine Learning</em></h3>

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)

<br/>

> 💡 *Predict exactly how many calories you burn during exercise — powered by your body metrics and workout data.*

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [📊 Dataset](#-dataset)
- [🏗️ Project Architecture](#️-project-architecture)
- [🔬 ML Pipeline](#-ml-pipeline)
- [📈 Model Performance](#-model-performance)
- [🚀 Sample Prediction](#-sample-prediction)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation & Setup](#️-installation--setup)
- [▶️ Usage](#️-usage)
- [🤝 Contributing](#-contributing)

---

## 🎯 Project Overview

**CalorIQ** is a machine learning project that uses **Linear Regression** to intelligently predict the number of calories burned during a workout session. By analyzing biometric and exercise data, CalorIQ gives you data-driven, personalized calorie estimates.

Whether you're a fitness enthusiast, a trainer, or a health-tech developer, CalorIQ provides accurate insights into your body's energy expenditure.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Smart Prediction** | Uses Linear Regression trained on real workout data |
| 🧹 **Data Cleaning** | Handles missing values and duplicate detection |
| 📉 **Visual Analysis** | Scatter plots for Calories vs Duration & Weight |
| 📏 **Multi-Metric Evaluation** | MAE, MSE, RMSE, and R² Score |
| 💾 **Model Persistence** | Saves the trained model with `joblib` |
| 📝 **Prediction History** | Logs every prediction with date, time & results |
| ☁️ **Google Colab Ready** | Runs seamlessly in Colab with no local setup |

---

## 📊 Dataset

The model is trained on a **calories dataset** (`calories.csv`) containing biometric and workout metrics.

### 🔢 Input Features

| Feature | Description | Unit |
|---|---|---|
| `Age` | Age of the individual | Years |
| `Height` | Height of the individual | Feet / cm |
| `Weight` | Body weight | kg |
| `Duration` | Workout duration | Minutes |
| `Heart_Rate` | Average heart rate during workout | BPM |
| `Body_Temp` | Body temperature during workout | °C |

### 🎯 Target Variable

| Variable | Description | Unit |
|---|---|---|
| `Calories` | Calories burned during the session | kcal |

---

## 🏗️ Project Architecture

```
CalorIQ
├── 📥  Data Ingestion       → Load calories.csv
│
├── 🔍  Exploratory Analysis  → Shape, columns, dtypes, stats
│
├── 🧹  Data Cleaning         → Drop nulls & duplicates
│
├── 📊  Visualization         → Scatter plots (Duration, Weight vs Calories)
│
├── ⚙️  Feature Engineering   → Select 6 input features, 1 target
│
├── ✂️  Train/Test Split      → 80% train | 20% test
│
├── 🤖  Model Training        → Linear Regression (sklearn)
│
├── 📈  Evaluation            → MAE, MSE, RMSE, R² Score
│
├── 🔮  Prediction            → Single or batch calorie estimates
│
└── 💾  Model Export          → Saved as .pki file via joblib
```

---

## 🔬 ML Pipeline

```python
# Step-by-step flow
1. Import Libraries       → pandas, numpy, matplotlib, sklearn
2. Load Dataset           → pd.read_csv("calories.csv")
3. Data Inspection        → .head(), .tail(), .info(), .describe()
4. Quality Check          → isnull().sum(), duplicated().sum()
5. Clean Data             → drop_duplicates()
6. Visualize Trends       → plt.scatter() — Duration & Weight vs Calories
7. Define Features (X)    → Age, Height, Weight, Duration, Heart_Rate, Body_Temp
8. Define Target (y)      → Calories
9. Split Data             → train_test_split(test_size=0.2, random_state=42)
10. Train Model           → LinearRegression().fit(X_train, y_train)
11. Predict               → model.predict(X_test)
12. Evaluate              → MAE, MSE, RMSE, R² Score
13. Save Model            → joblib.dump(model, "model.pki")
```

---

## 📈 Model Performance

The trained Linear Regression model achieved **excellent** results:

```
┌────────────────────────────────────────────────────┐
│              📊  Model Evaluation Results           │
├──────────────────────────────┬─────────────────────┤
│  Mean Absolute Error (MAE)   │      8.44 kcal       │
│  Mean Squared Error (MSE)    │     132.03 kcal²     │
│  Root Mean Squared Error     │     11.49 kcal       │
│  R² Score                   │      0.9673  ✅       │
└──────────────────────────────┴─────────────────────┘
```

> ✅ **R² Score of ~0.97** means the model explains **96.7%** of the variance in calorie burn — a highly accurate prediction!

---

## 🚀 Sample Prediction

Here's a real example from CalorIQ:

```python
# Input — A 30-year-old person's workout stats
Age        = 30
Weight     = 75 kg
Height     = 5.5 ft
Duration   = 60 mins
Heart_Rate = 70 BPM
Body_Temp  = 36.5 °C

# Output
🔥 CALORIES BURNED = 410.41 kcal
```

### 📋 Prediction History Sample

| Run | Date | Time | Age | Height | Weight | Duration | Heart Rate | Body Temp | Predicted kcal |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 19 Oct 2026 | 10:48:56 AM | 30 | 177.0 cm | 93.0 kg | 46 min | 108 BPM | 40.0 °C | **313.91** |
| 2 | 19 Oct 2026 | 10:49:28 AM | 30 | 177.0 cm | 93.0 kg | 46 min | 69 BPM | 40.0 °C | **236.29** |
| 3 | 19 Oct 2026 | 10:49:48 AM | 30 | 177.0 cm | 162.0 kg | 46 min | 48 BPM | 40.0 °C | **209.91** |

---

## 📁 Project Structure

```
📦 CalorIQ/
│
├── 📓 CalorIQ.ipynb                             # Main Jupyter Notebook
├── 📄 calories.csv                              # Training dataset
├── 💾 mobile_charging_time_prediction_model.pki # Saved trained model
├── 📊 caloriq_prediction_history.csv            # Prediction log file
├── 📑 CalorIQ_Presentation_v3.pptx              # Project presentation
└── 📘 README.md                                 # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites

Make sure you have Python 3.x installed. Then install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### Running on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `CalorIQ.ipynb`
3. Upload `calories.csv` to the Colab session
4. Run all cells — `Runtime → Run All`

### Running Locally

```bash
# Clone or download the project
cd CalorIQ

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook CalorIQ.ipynb
```

---

## ▶️ Usage

### Making a Custom Prediction

Edit these variables in the notebook to predict your own calorie burn:

```python
Age        = 25       # Your age in years
Weight     = 70       # Your weight in kg
Height     = 5.8      # Your height in feet
Duration   = 45       # Your workout duration in minutes
Heart_Rate = 85       # Average heart rate in BPM
Body_Temp  = 37.0     # Body temperature in °C

# Run the prediction cell and get:
# 🔥 CALORIES BURNED = <your_result> kcal
```

### Loading the Saved Model

```python
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("Calories.pki")

# Prepare your input
new_data = pd.DataFrame({
    'Age': [25], 'Height': [5.8], 'Weight': [70],
    'Duration': [45], 'Heart_Rate': [85], 'Body_Temp': [37.0]
})

# Get prediction
calories = model.predict(new_data)
print(f"🔥 Calories Burned: {round(calories[0], 2)} kcal")
```

---

## 🛠️ Technologies Used

<div align="center">

| Tool | Purpose |
|---|---|
| 🐍 **Python 3.12** | Core programming language |
| 📓 **Google Colab** | Cloud notebook environment |
| 🐼 **Pandas** | Data loading & manipulation |
| 🔢 **NumPy** | Numerical computing |
| 📊 **Matplotlib** | Data visualization |
| 🤖 **Scikit-learn 1.6.1** | ML model (Linear Regression) |
| 💾 **Joblib** | Model serialization |

</div>

---

## 🤝 Contributing

Contributions are welcome! Here's how you can improve CalorIQ:

1. 🍴 Fork this repository
2. 🌿 Create your feature branch: `git checkout -b feature/improve-model`
3. 💾 Commit your changes: `git commit -m "Add Random Forest model"`
4. 📤 Push to the branch: `git push origin feature/improve-model`
5. 🔁 Open a Pull Request

### 💡 Ideas for Enhancement

- [ ] Try **Random Forest** or **XGBoost** for better accuracy
- [ ] Add **gender** as an additional feature
- [ ] Build a **Streamlit web app** for interactive predictions
- [ ] Add **cross-validation** for more robust evaluation
- [ ] Include **feature importance** visualization

---

<div align="center">

---

Made with ❤️ and 🔥 by the CalorIQ Team

*Predict Smarter. Train Better. Burn Accurately.*

</div>
