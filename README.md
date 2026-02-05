FastRaces🏎️

FastRaces is a data-driven project dedicated to predicting the outcomes of Formula 1 races. By harnessing the full historical scope of data available through the FastF1 Python package, this project aims to build a highly accurate predictive model. The core philosophy is that by engineering a rich and comprehensive feature set, we can train a model to understand the deep, nuanced factors that determine a race's final classification.

🚀 Key Features

1. Comprehensive Data Collection: Gathers data across multiple F1 seasons (2019-2023) using the fastf1 library.

2. Rich Feature Engineering: The dataset isn't just lap times. It includes:

3. Qualifying Performance: Grid Position and time delta to the pole-sitter.

4. Historical Performance: A driver's (or their team's) finishing position at the same track in the previous year.

5. Intelligent Rookie Handling: Uses a "Team-Based Fallback" to estimate the performance of rookie drivers based on their car's potential.

6. Detailed Weather Data: Captures Air Temperature, Track Temperature, Humidity, Wind Speed, Wind Direction, and a Rainfall flag.

7. Race-Specific Stats: Includes the number of pit stops for each driver.

8. Predictive Modeling: Utilizes an XGBoost model, a powerful gradient-boosting algorithm perfect for structured, tabular data like ours.

🛠️ Tech Stack

Language: Python 3.9+

Core Libraries:

FastF1: The engine for accessing F1 timing, telemetry, and session data.

Pandas: For data manipulation and analysis.

Scikit-learn: For data preprocessing and model evaluation.

XGBoost: The machine learning model.

Matplotlib/Seaborn: For data visualization (TBD).

📊 Usage

The project is broken down into two main steps: data collection and model training.

Step 1: Data Collection

Run the data builder script to fetch and compile the comprehensive dataset. Expected runtime is long (30+ minutes).

⚠️ IMPORTANT: This script downloads a large amount of data spanning multiple years. Due to API rate limits imposed by the F1 data sources, the script has built-in delays (time.sleep()).

Do not remove the delays, or your IP may be temporarily blocked for making too many requests.

The script will create a file named f1_race_data_w_weather.csv.

Step 2: Training the Model

Once the dataset is created, you can train the prediction model. This will process the raw data, train the XGBoost regressor, and save the trained model to a file for later use.

🧠 Methodology

The core philosophy of this project is that feature engineering is paramount. While a complex model is powerful, its performance is capped by the quality of the data it receives.

Our approach includes:

1. Rolling Historical Data: To predict a season (e.g., 2023), we use the previous season (2022) as a historical benchmark for driver and team performance at each track.

2. Team-Based Fallback for Rookies: Instead of penalizing a rookie driver with a default "last place" history, we infer their car's potential by using the average performance of their team at that track in the previous year. This provides a much more realistic baseline.

3. Contextual Features: We enrich the dataset with weather and pit stop data, allowing the model to learn complex relationships, such as how rain can neutralize a car's raw pace advantage or how a different strategy can affect the outcome.

4. Hyperparameter Tuning: Optimize the XGBoost model's parameters using GridSearchCV.

🙏 Acknowledgements: 

This project would not be possible without the incredible work done by the developers of the FastF1 library.
