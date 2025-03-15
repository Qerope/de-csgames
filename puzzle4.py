import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Load the dataset
def load_data(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

# Process the game data into features
def process_game(game):
    features = {}
    
    if 'cards' not in game or not isinstance(game['cards'], list):
        raise ValueError("Invalid game data: missing or malformed cards.")
    
    for card in game["cards"]:
        suit_key = f"suit_{card['suit']}"
        rank_key = f"rank_{card['rank']}"
        
        features[suit_key] = features.get(suit_key, 0) + 1
        features[rank_key] = features.get(rank_key, 0) + 1
    
    # Use cumulative sum for actions to capture trends
    actions = game.get("actions", [])
    for i in range(len(actions)):
        features[f"action_{i}"] = sum(actions[:i+1])  # Cumulative sum
    
    return features

# Prepare the dataset
def prepare_dataset(data):
    processed_data = [process_game(game) for game in data if 'outcome' in game]
    df = pd.DataFrame(processed_data).fillna(0)
    df["outcome"] = [game["outcome"] for game in data if 'outcome' in game]
    return df

# Train the model with multiple approaches and hyperparameter optimization
def train_multiple_models(data):
    df = prepare_dataset(data)
    X = df.drop(columns=["outcome"])
    y = df["outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = X.columns
    numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
    
    # Define the models to try
    models = {
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    # Define hyperparameter grids with broader ranges
    param_grids = {
        "GradientBoosting": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.5],
            "max_depth": [3, 5, 7, 10, 15],
            "subsample": [0.5, 0.7, 1.0],  # Added subsample for randomness
            "min_samples_split": [2, 5, 10]
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False]  # Added bootstrap parameter
        },
        "SVR": {
            "C": [1, 10, 100, 1000],
            "epsilon": [0.01, 0.1, 0.2, 0.5],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto"]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "max_depth": [3, 5, 7, 10],
            "subsample": [0.5, 0.7, 1.0],
            "colsample_bytree": [0.5, 0.7, 1.0],
            "min_child_weight": [1, 5, 10]
        }
    }
    
    best_model = None
    best_mse = float("inf")
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Perform hyperparameter tuning using RandomizedSearchCV
        model_search = RandomizedSearchCV(model, param_grids[model_name], cv=3, n_iter=20, scoring="neg_mean_squared_error", random_state=42)
        
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model_search)])
        pipeline.fit(X_train, y_train)
        
        # Get the predictions and calculate MSE
        predictions = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        print(f"{model_name} MSE: {mse:.2f}")
        
        # If the current model has a better MSE, save it
        if mse < best_mse:
            best_mse = mse
            best_model = pipeline
            
    print(f"Best model: {best_model.named_steps['regressor'].best_estimator_} with MSE: {best_mse:.2f}")
    
    return best_model

# Predict the optimal actions for a new game
def predict_optimal_actions(game, model, X_columns):
    features = process_game(game)
    df = pd.DataFrame([features]).fillna(0)
    df = df.reindex(columns=X_columns, fill_value=0)
    return model.predict(df)[0]

# Example usage
data = load_data("10k_games.json")
best_model = train_multiple_models(data)

# Predict on a new game example
new_game = {
    "cards": [
        {"name": "ACE_OF_SPADES", "suit": "SPADES", "rank": 1},
        {"name": "TWO_OF_HEARTS", "suit": "HEARTS", "rank": 2}
    ],
    "actions": [0] * 15
}

# Assuming the model is trained and selected already
print("Predicted optimal outcome:", predict_optimal_actions(new_game, best_model, best_model.named_steps['preprocessor'].transformers_[0][2]))
