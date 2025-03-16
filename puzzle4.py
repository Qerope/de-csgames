import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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

# Create the initial XGBoost parameter grid
def create_xgb_param_grid():
    return {
        "n_estimators": [50, 100, 200, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 6, 7, 10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.5, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [0.1, 0.5, 1, 2],
        "scale_pos_weight": [1, 2, 5]
    }

# Dynamically adjust the parameter grid based on previous results
def adjust_param_grid(current_grid, best_params, previous_results):
    """
    Dynamically adjust the hyperparameter grid based on previous results
    """
    if not previous_results:
        return current_grid  # If no previous results, return the initial grid
    
    # Focus the search on the best performing parameters
    new_grid = current_grid.copy()
    
    # Example: Narrow the search space around the best performing learning_rate
    if 'learning_rate' in best_params:
        learning_rate_best = best_params['learning_rate']
        new_grid['learning_rate'] = [max(0.0001, learning_rate_best - 0.05), 
                                      max(0.0001, learning_rate_best - 0.02), 
                                      learning_rate_best, 
                                      learning_rate_best + 0.02, 
                                      learning_rate_best + 0.05]
    
    # Similarly, adjust other parameters like n_estimators, max_depth, etc., based on results
    if 'n_estimators' in best_params:
        n_estimators_best = best_params['n_estimators']
        new_grid['n_estimators'] = [max(50, n_estimators_best - 100), 
                                     n_estimators_best, 
                                     n_estimators_best + 100]

    if 'max_depth' in best_params:
        max_depth_best = best_params['max_depth']
        new_grid['max_depth'] = [max(3, max_depth_best - 2), 
                                 max_depth_best, 
                                 max_depth_best + 2]
    
    return new_grid

# Function to train and adjust XGBoost using adaptive search
def train_with_dynamic_search(X_train, y_train, X_test, y_test, model, initial_param_grid, n_iter=50):
    # Store results and best parameters
    previous_results = []
    best_mse = float("inf")
    best_params = None
    
    for i in range(n_iter):
        print(f"Iteration {i+1}/{n_iter}...")
        
        # Perform RandomizedSearchCV with current parameter grid
        model_search = RandomizedSearchCV(model, initial_param_grid, n_iter=20, scoring="neg_mean_squared_error", random_state=42)
        model_search.fit(X_train, y_train)
        
        # Get the best hyperparameters and model performance
        current_best_params = model_search.best_params_
        current_best_mse = -model_search.best_score_  # RandomizedSearchCV returns negative MSE
        
        # Store the best parameters and MSE if they improve
        if current_best_mse < best_mse:
            best_mse = current_best_mse
            best_params = current_best_params
        
        # Print the current best result
        print(f"Best MSE so far: {best_mse:.2f} with params: {best_params}")
        
        # Adjust the parameter grid based on current results
        initial_param_grid = adjust_param_grid(initial_param_grid, best_params, previous_results)
        
        # Store current results for future adjustment
        previous_results.append({
            "params": current_best_params,
            "mse": current_best_mse
        })
    
    return best_params, best_mse

# Function to train the XGBoost model
def train_xgboost_model(data):
    df = prepare_dataset(data)
    X = df.drop(columns=["outcome"])
    y = df["outcome"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initializing the preprocessor
    numeric_features = X.columns
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
    
    # Create the initial XGBoost parameter grid
    initial_param_grid = create_xgb_param_grid()
    
    # Train the model with dynamic search
    model = XGBRegressor(random_state=42)
    best_params, best_mse = train_with_dynamic_search(X_train, y_train, X_test, y_test, model, initial_param_grid, n_iter=50)
    
    print(f"Best XGBoost model parameters: {best_params}")
    print(f"Best MSE: {best_mse:.2f}")
    
    return model, best_params

# Predict the optimal actions for a new game
def predict_optimal_actions(game, model, X_columns):
    features = process_game(game)
    df = pd.DataFrame([features]).fillna(0)
    df = df.reindex(columns=X_columns, fill_value=0)
    return model.predict(df)[0]

# Example usage
data = load_data("10k_games.json")
model, best_params = train_xgboost_model(data)

# Predict on a new game example
new_game = {
    "cards": [
        {"name": "ACE_OF_SPADES", "suit": "SPADES", "rank": 1},
        {"name": "TWO_OF_HEARTS", "suit": "HEARTS", "rank": 2}
    ],
    "actions": [0] * 15
}

# Assuming the model is trained and selected already
print("Predicted optimal outcome:", predict_optimal_actions(new_game, model, model.get_booster().feature_names))
