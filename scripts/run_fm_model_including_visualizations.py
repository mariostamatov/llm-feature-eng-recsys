#!/usr/bin/env python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import warnings
import random
import os
import optuna
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Define output directory for visualizations
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VIZ_DIR = os.path.join(PROJECT_ROOT, 'visualizations')
os.makedirs(VIZ_DIR, exist_ok=True)

def set_seed(seed: int = 42):
    """Set the seed for reproducibility in python, numpy and torch."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print('DataLoaders created successfully.')

def create_feature_matrix(df, keyword_column):
    # Convert user_id and movie_id to one-hot encoded features
    user_features = pd.get_dummies(df['user_id_cat'], prefix='user')
    movie_features = pd.get_dummies(df['movie_id_cat'], prefix='movie')

    # Use CountVectorizer for keywords
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    keyword_features = vectorizer.fit_transform(df[keyword_column])

    # Combine all features
    X = hstack([user_features, movie_features, keyword_features])
    return X, vectorizer

class FactorizationMachine(nn.Module):
    def __init__(self, num_features, embedding_dim=10):
        super(FactorizationMachine, self).__init__()
        self.bias = nn.Parameter(torch.randn(1))
        self.weights = nn.Linear(num_features, 1, bias=False)
        self.embeddings = nn.Linear(num_features, embedding_dim, bias=False)
    def forward(self, x, return_components=False):
        linear_part = self.bias + self.weights(x)
        embedded_x = self.embeddings(x)
        
        # More numerically stable way to compute the interaction term
        sum_of_squares = embedded_x.pow(2).sum(1, keepdim=True)
        square_of_sum = self.embeddings(x).sum(1, keepdim=True).pow(2)
        
        interaction_part = 0.5 * (square_of_sum - sum_of_squares)
        
        if return_components:
            return linear_part, interaction_part

        return linear_part + interaction_part

def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for features, targets in train_loader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(train_loader.dataset)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            predictions = model(features)
            loss = criterion(predictions, targets)
            total_loss += loss.item() * len(targets)
    mse = total_loss / len(test_loader.dataset)
    return np.sqrt(mse)

def bootstrap_rmse(model, test_dataset, criterion, device, n_bootstraps=10000):
    bootstrapped_rmses = []
    dataset_size = len(test_dataset)
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping RMSE"):
        # Resample with replacement
        indices = np.random.choice(dataset_size, dataset_size, replace=True)
        bootstrapped_subset = torch.utils.data.Subset(test_dataset, indices)
        bootstrapped_loader = DataLoader(bootstrapped_subset, batch_size=1024, shuffle=False)
        rmse = evaluate_model(model, bootstrapped_loader, criterion, device)
        bootstrapped_rmses.append(rmse)
    return np.array(bootstrapped_rmses)

def run_experiment(train_loader, test_loader, test_dataset, num_features, num_epochs, learning_rate=0.01, embedding_dim=50, weight_decay=0, optimizer_name='Adam', run_bootstrap=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = FactorizationMachine(num_features=num_features, embedding_dim=embedding_dim).to(device)
    criterion = nn.MSELoss()
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_rmse_history = []
    test_rmse_history = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_model(model, train_loader, optimizer, criterion, device)
        train_rmse = evaluate_model(model, train_loader, criterion, device)
        test_rmse = evaluate_model(model, test_loader, criterion, device)
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)
        print(f"Epoch {epoch+1}/{num_epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    
    ci_95 = (0, 0)
    ci_99 = (0, 0)
    if run_bootstrap:
        # Perform bootstrapping for confidence interval
        bootstrapped_rmses = bootstrap_rmse(model, test_dataset, criterion, device)
        
        # 95% CI
        lower_bound_95 = np.percentile(bootstrapped_rmses, 2.5)
        upper_bound_95 = np.percentile(bootstrapped_rmses, 97.5)
        ci_95 = (lower_bound_95, upper_bound_95)

        # 99% CI
        lower_bound_99 = np.percentile(bootstrapped_rmses, 0.5)
        upper_bound_99 = np.percentile(bootstrapped_rmses, 99.5)
        ci_99 = (lower_bound_99, upper_bound_99)
    
    final_train_rmse = evaluate_model(model, train_loader, criterion, device)
    final_test_rmse = evaluate_model(model, test_loader, criterion, device)

    return final_train_rmse, final_test_rmse, ci_95, ci_99, model, train_rmse_history, test_rmse_history

def objective(trial, X, y, num_features):
    # Define hyperparameters to tune
    embedding_dim = trial.suggest_int('embedding_dim', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    num_epochs = trial.suggest_int('num_epochs', 5, 50) # Allow Optuna to tune the number of epochs
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    batch_size = trial.suggest_int('batch_size', 32, 2048, log=True)

    train_loader, test_loader, test_dataset = create_dataloaders(X, y, batch_size=batch_size)

    # Run the experiment with the suggested hyperparameters
    _, test_rmse, _, _, _, _, _ = run_experiment(
        train_loader,
        test_loader,
        test_dataset,
        num_features,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        embedding_dim=embedding_dim,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        run_bootstrap=False # Disable bootstrapping during tuning for speed
    )
    return test_rmse


def create_dataloaders(X, y, batch_size, test_size=0.2, random_state=42):
    indices = np.arange(X.shape[0])
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    X_train, X_test = X.tocsr()[train_indices], X.tocsr()[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    X_train_tensor = torch.from_numpy(X_train.toarray()).float()
    X_test_tensor = torch.from_numpy(X_test.toarray()).float()
    y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
    y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, test_dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def inspect_linear_weights(model, df, vectorizer, model_type):
    print(f"Linear Weights Inspection for {model_type} Model")
    weights = model.weights.weight.detach().cpu().numpy().flatten()

    num_users = df['user_id_cat'].nunique()
    num_movies = df['movie_id_cat'].nunique()
    num_keywords = len(vectorizer.vocabulary_)

    # User weights
    user_weights = weights[0:num_users]
    avg_user_weight = np.mean(np.abs(user_weights))
    print(f"Average absolute user weight: {avg_user_weight:.4f}")

    # Movie weights
    movie_weights = weights[num_users:num_users + num_movies]
    avg_movie_weight = np.mean(np.abs(movie_weights))
    print(f"Average absolute movie weight: {avg_movie_weight:.4f}")

    # Keyword weights
    keyword_weights = weights[num_users + num_movies:num_users + num_movies + num_keywords]
    avg_keyword_weight = np.mean(np.abs(keyword_weights))
    print(f"Average absolute keyword weight: {avg_keyword_weight:.4f}")

    # Compare
    max_avg_weight = max(avg_user_weight, avg_movie_weight, avg_keyword_weight)
    if max_avg_weight == avg_user_weight:
        print("User features appear to have the highest average linear weight.")
    elif max_avg_weight == avg_movie_weight:
        print("Movie features appear to have the highest average linear weight.")
    else:
        print("Keyword features appear to have the highest average linear weight.")

    print("Note: This only considers the linear part of the FM. The factorization part (embeddings) also contributes significantly.")

def analyze_prediction_contribution(model, test_loader, device):
    model.eval()
    total_linear_contribution = 0
    total_interaction_contribution = 0
    total_samples = 0

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            linear_part, interaction_part = model(features, return_components=True)
            
            total_linear_contribution += torch.abs(linear_part).sum().item()
            total_interaction_contribution += torch.abs(interaction_part).sum().item()
            total_samples += len(features)

    avg_linear_contribution = total_linear_contribution / total_samples
    avg_interaction_contribution = total_interaction_contribution / total_samples
    
    total_contribution = avg_linear_contribution + avg_interaction_contribution
    
    percent_linear = (avg_linear_contribution / total_contribution) * 100
    percent_interaction = (avg_interaction_contribution / total_contribution) * 100

    return percent_linear, percent_interaction

def plot_error_distribution(model_human, model_llm, loader_human, loader_llm, device):
    """Generates and saves histograms of the prediction errors for both models."""
    print("Generating error distribution plots...")
    model_human.eval()
    model_llm.eval()
    
    errors_human = []
    with torch.no_grad():
        for features, targets in loader_human:
            features, targets = features.to(device), targets.to(device)
            predictions = model_human(features)
            errors_human.extend((predictions - targets).cpu().numpy().flatten())

    errors_llm = []
    with torch.no_grad():
        for features, targets in loader_llm:
            features, targets = features.to(device), targets.to(device)
            predictions = model_llm(features)
            errors_llm.extend((predictions - targets).cpu().numpy().flatten())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    ax1.hist(errors_human, bins=50, alpha=0.7, label='Errors')
    ax1.axvline(x=0, color='r', linestyle='--')
    ax1.set_title('Prediction Error Distribution (Human Model)')
    ax1.set_xlabel('Prediction Error (Predicted - Actual)')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2.hist(errors_llm, bins=50, alpha=0.7, label='Errors', color='lightgreen')
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title('Prediction Error Distribution (LLM Model)')
    ax2.set_xlabel('Prediction Error (Predicted - Actual)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'error_distribution.png'))
    print(f"Error distribution plot saved to {os.path.join(VIZ_DIR, 'error_distribution.png')}")

def plot_keyword_weights(model_human, model_llm, df, vectorizer_human, vectorizer_llm):
    """Generates and saves histograms of the keyword linear weights for both models."""
    print("Generating keyword weight distribution plots...")
    
    # Extract Human Keyword Weights
    weights_human = model_human.weights.weight.detach().cpu().numpy().flatten()
    num_users_human = df['user_id_cat'].nunique()
    num_movies_human = df['movie_id_cat'].nunique()
    keyword_weights_human = weights_human[num_users_human + num_movies_human:]

    # Extract LLM Keyword Weights
    weights_llm = model_llm.weights.weight.detach().cpu().numpy().flatten()
    num_users_llm = df['user_id_cat'].nunique()
    num_movies_llm = df['movie_id_cat'].nunique()
    keyword_weights_llm = weights_llm[num_users_llm + num_movies_llm:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    ax1.hist(keyword_weights_human, bins=50, alpha=0.7)
    ax1.set_title('Keyword Linear Weight Distribution (Human Model)')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')

    ax2.hist(keyword_weights_llm, bins=50, alpha=0.7, color='lightgreen')
    ax2.set_title('Keyword Linear Weight Distribution (LLM Model)')
    ax2.set_xlabel('Weight Value')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'keyword_weights.png'))
    print(f"Keyword weight distribution plot saved to {os.path.join(VIZ_DIR, 'keyword_weights.png')}")

def plot_learning_curves(history_human, history_llm):
    """Generates and saves learning curve plots for both models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # Human Model Learning Curve
    ax1.plot(history_human['train'], label='Train RMSE')
    ax1.plot(history_human['test'], label='Test RMSE')
    ax1.set_title('Learning Curve (Human Model)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(True)

    # LLM Model Learning Curve
    ax2.plot(history_llm['train'], label='Train RMSE')
    ax2.plot(history_llm['test'], label='Test RMSE')
    ax2.set_title('Learning Curve (LLM Model)')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'learning_curves.png'))
    print(f"Learning curves plot saved to {os.path.join(VIZ_DIR, 'learning_curves.png')}")

def plot_rmse_comparison(human_rmse, llm_rmse, human_ci_95, llm_ci_95, human_ci_99, llm_ci_99):
    """Generates and saves a bar chart comparing the RMSE of the two models with 95% and 99% CIs."""
    labels = ['Human Keywords', 'LLM Keywords']
    x_pos = np.arange(len(labels))
    rmses = [human_rmse, llm_rmse]
    
    # CIs need to be transformed into a format suitable for errorbar plotting (distance from the mean)
    ci_95 = np.array([
        [human_rmse - human_ci_95[0], llm_rmse - llm_ci_95[0]],
        [human_ci_95[1] - human_rmse, llm_ci_95[1] - llm_rmse]
    ])
    ci_99 = np.array([
        [human_rmse - human_ci_99[0], llm_rmse - llm_ci_99[0]],
        [human_ci_99[1] - human_rmse, llm_ci_99[1] - llm_rmse]
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(x_pos, rmses, yerr=ci_95, align='center', alpha=0.7, ecolor='black', capsize=10, color=['skyblue', 'lightgreen'], label='95% CI')
    # Plot 99% CI with a different style
    ax.errorbar(x_pos, rmses, yerr=ci_99, fmt='none', ecolor='red', capsize=5, label='99% CI')

    ax.set_ylabel('Root Mean Squared Error (RMSE)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('Model Performance Comparison: Human vs. LLM Keywords')
    ax.yaxis.grid(True)
    ax.legend()

    # Add RMSE values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center') 

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'rmse_comparison.png'))
    print(f"RMSE comparison plot saved to {os.path.join(VIZ_DIR, 'rmse_comparison.png')}")

def visualize_embeddings_pca(model, df, vectorizer, model_type, num_movies_to_plot=200):
    """Generates and saves a PCA plot of the movie embeddings, colored by genre."""
    print(f"Generating PCA plot for {model_type} model...")

    # Extract movie embeddings
    num_users = df['user_id_cat'].nunique()
    num_movies = df['movie_id_cat'].nunique()
    movie_embeddings = model.embeddings.weight.T.detach().cpu().numpy()[num_users:num_users + num_movies]

    # Ensure we don't plot more movies than we have
    if num_movies < num_movies_to_plot:
        num_movies_to_plot = num_movies

    # Get a random sample of movies to plot
    movie_indices = np.random.choice(num_movies, num_movies_to_plot, replace=False)
    sampled_embeddings = movie_embeddings[movie_indices]

    # Get movie titles and genres for the sampled movies
    movie_id_map = df[['movie_id_cat', 'title', 'primary_genre']].drop_duplicates().set_index('movie_id_cat')
    sampled_info = movie_id_map.loc[movie_indices]
    sampled_titles = sampled_info['title'].values
    sampled_genres = sampled_info['primary_genre'].values

    # PCA transformation
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(sampled_embeddings)

    # Create plot
    plt.figure(figsize=(16, 12))
    
    # Define a color map for top genres
    top_genres = df['primary_genre'].value_counts().nlargest(10).index
    colors = plt.cm.get_cmap('tab10', len(top_genres))
    genre_color_map = {genre: colors(i) for i, genre in enumerate(top_genres)}
    default_color = 'lightgrey'

    # Create scatter plot with colors
    for i, genre in enumerate(sampled_genres):
        color = genre_color_map.get(genre, default_color)
        plt.scatter(pca_results[i, 0], pca_results[i, 1], color=color, label=genre if genre in genre_color_map and i == np.where(sampled_genres == genre)[0][0] else "")

    # Add labels to a subset of points
    for i in range(0, len(sampled_titles), 10): # Label every 10th point
        plt.text(pca_results[i, 0], pca_results[i, 1], sampled_titles[i], fontsize=9)

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=genre, markersize=10, markerfacecolor=color) for genre, color in genre_color_map.items()]
    handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Other', markersize=10, markerfacecolor=default_color))
    plt.legend(title='Primary Genre', handles=handles)

    plt.title(f'PCA Visualization of Movie Embeddings by Genre ({model_type} Model)')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.grid(True)
    output_path = os.path.join(VIZ_DIR, f'{model_type.lower()}_embeddings_pca_colored.png')
    plt.savefig(output_path)
    print(f"PCA plot for {model_type} model saved to {output_path}")

def main():
    set_seed(42)
    print('Loading and Preprocessing Data')
    DATA_PATH = 'https://github.com/mariostamatov/llm-feature-eng-recsys/raw/main/data/final_llm_features_dataset.parquet'
    MOVIES_PATH = 'https://raw.githubusercontent.com/mariostamatov/llm-feature-eng-recsys/refs/heads/main/data/ml-25m/movies.csv'

    try:
        df = pd.read_parquet(DATA_PATH)
        print(f"Real dataset loaded successfully. Shape: {df.shape}")

        # Load and merge genre data
        movies_df = pd.read_csv(MOVIES_PATH)
        df = pd.merge(df, movies_df[['movieId', 'title', 'genres']], left_on='movie_id', right_on='movieId', how='left')
        df.rename(columns={'title_x': 'title'}, inplace=True)
        df['primary_genre'] = df['genres'].apply(lambda x: x.split('|')[0] if isinstance(x, str) else 'Unknown')
        df.drop(columns=['movieId', 'title_y', 'genres'], inplace=True, errors='ignore')
        print("Genre data loaded and merged successfully.")

    except FileNotFoundError:
        print(f"Warning: The data file was not found at {DATA_PATH} or {MOVIES_PATH}")
        print('Using a dummy dataframe for demonstration purposes.')
        df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            'movie_id': [101, 102, 101, 103, 102, 104, 103, 104, 105, 106, 105, 106, 107, 108, 107, 108],
            'rating': [5, 3, 4, 2, 5, 4, 3, 5, 2, 4, 3, 5, 4, 3, 2, 5],
            'title': ['Movie 101', 'Movie 102', 'Movie 101', 'Movie 103', 'Movie 102', 'Movie 104', 'Movie 103', 'Movie 104', 'Movie 105', 'Movie 106', 'Movie 105', 'Movie 106', 'Movie 107', 'Movie 108', 'Movie 107', 'Movie 108'],
            'human_keywords': ['action, thriller', 'comedy, romance', 'action, thriller', 'sci-fi, adventure', 'comedy, romance', 'drama', 'sci-fi, adventure', 'drama', 'mystery', 'fantasy', 'mystery', 'fantasy', 'crime', 'history', 'crime', 'history'],
            'llm_keywords': ['fast-paced, explosive', 'lighthearted, love', 'fast-paced, explosive', 'space, future', 'lighthearted, love', 'emotional, serious', 'space, future', 'emotional, serious', 'suspenseful, investigation', 'magical, mythical', 'suspenseful, investigation', 'magical, mythical', 'gritty, investigation', 'period piece, factual', 'gritty, investigation', 'period piece, factual'],
            'genres': ['Action|Thriller', 'Comedy|Romance', 'Action|Thriller', 'Sci-Fi|Adventure', 'Comedy|Romance', 'Drama', 'Sci-Fi|Adventure', 'Drama', 'Mystery', 'Fantasy', 'Mystery', 'Fantasy', 'Crime', 'History', 'Crime', 'History']
        })
        df['primary_genre'] = df['genres'].apply(lambda x: x.split('|')[0])


    print('Performing Feature Engineering')
    df['user_id_cat'] = df['user_id'].astype('category').cat.codes
    df['movie_id_cat'] = df['movie_id'].astype('category').cat.codes

    X_human, vectorizer_human = create_feature_matrix(df.copy(), 'human_keywords')
    X_llm, vectorizer_llm = create_feature_matrix(df.copy(), 'llm_keywords')
    y = df['rating'].values

    print(f"Control (Human) Feature Matrix Shape: {X_human.shape}")
    print(f"Experimental (LLM) Feature Matrix Shape: {X_llm.shape}")

    # These will be created inside the objective function with the suggested batch size
    train_loader_human, test_loader_human, test_dataset_human = None, None, None
    train_loader_llm, test_loader_llm, test_dataset_llm = None, None, None
    print('DataLoaders created successfully.')

    print('Starting Hyperparameter Tuning for Human Model')
    study_human = optuna.create_study(direction='minimize')
    study_human.optimize(lambda trial: objective(trial, X_human, y, X_human.shape[1]), n_trials=100)
    best_params_human = study_human.best_trial.params

    print('Starting Hyperparameter Tuning for LLM Model')
    study_llm = optuna.create_study(direction='minimize')
    study_llm.optimize(lambda trial: objective(trial, X_llm, y, X_llm.shape[1]), n_trials=100)
    best_params_llm = study_llm.best_trial.params

    print('Starting Final Control Group Experiment (Human Keywords) with Best Hyperparameters')
    num_features_human = X_human.shape[1]
    train_loader_human, test_loader_human, test_dataset_human = create_dataloaders(X_human, y, batch_size=best_params_human['batch_size'])
    train_rmse_human, test_rmse_human, ci_95_human, ci_99_human, model_human, train_rmse_history_human, test_rmse_history_human = run_experiment(
        train_loader_human,         test_loader_human,         test_dataset_human,         num_features_human,         num_epochs=best_params_human['num_epochs'],
        learning_rate=best_params_human['learning_rate'],         embedding_dim=best_params_human['embedding_dim'],        weight_decay=best_params_human['weight_decay'],
        optimizer_name=best_params_human['optimizer']
    )
    print(f"Final Train RMSE for Control (Human) Model: {train_rmse_human:.4f}")
    print(f"Final Test RMSE for Control (Human) Model: {test_rmse_human:.4f}")

    print('Starting Final Experimental Group Experiment (LLM Keywords) with Best Hyperparameters')
    num_features_llm = X_llm.shape[1]
    train_loader_llm, test_loader_llm, test_dataset_llm = create_dataloaders(X_llm, y, batch_size=best_params_llm['batch_size'])
    train_rmse_llm, test_rmse_llm, ci_95_llm, ci_99_llm, model_llm, train_rmse_history_llm, test_rmse_history_llm = run_experiment(
        train_loader_llm,         test_loader_llm,         test_dataset_llm,         num_features_llm,         num_epochs=best_params_llm['num_epochs'],
        learning_rate=best_params_llm['learning_rate'],         embedding_dim=best_params_llm['embedding_dim'],        weight_decay=best_params_llm['weight_decay'],
        optimizer_name=best_params_llm['optimizer']
    )

    print('Inspecting Linear Weights')
    inspect_linear_weights(model_human, df, vectorizer_human, "Human")
    inspect_linear_weights(model_llm, df, vectorizer_llm, "LLM")

    print('Analyzing Prediction Contributions')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear_human, interaction_human = analyze_prediction_contribution(model_human, test_loader_human, device)
    linear_llm, interaction_llm = analyze_prediction_contribution(model_llm, test_loader_llm, device)

    
    print('EXPERIMENT RESULTS')
    print(f"Train RMSE (Human Keywords): {train_rmse_human:.4f}")
    print(f"Test RMSE (Human Keywords): {test_rmse_human:.4f} (95% CI: {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f}) (99% CI: {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f})")
    print(f"  Contribution -> Linear: {linear_human:.2f}%, Interaction: {interaction_human:.2f}%")
    print(f"Train RMSE (LLM Keywords):   {train_rmse_llm:.4f}")
    print(f"Test RMSE (LLM Keywords):   {test_rmse_llm:.4f} (95% CI: {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f}) (99% CI: {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f})")
    print(f"  Contribution -> Linear: {linear_llm:.2f}%, Interaction: {interaction_llm:.2f}%")

    print('\nBest Hyperparameters Found:')
    print("Human Model:")
    for key, value in best_params_human.items():
        print(f"    {key}: {value}")
    print("LLM Model:")
    for key, value in best_params_llm.items():
        print(f"    {key}: {value}")
    
    if ci_95_llm[1] < ci_95_human[0]:
        print(f"Hypothesis Confirmed (at 95% confidence): LLM-based model performed statistically significantly better (95% CI: {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f} vs {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f}).")
    elif ci_95_human[1] < ci_95_llm[0]:
        print(f"Hypothesis Rejected (at 95% confidence): Human-based model performed statistically significantly better (95% CI: {ci_95_human[0]:.4f}-{ci_95_human[1]:.4f} vs {ci_95_llm[0]:.4f}-{ci_95_llm[1]:.4f}).")
    else:
        print('Result: No statistically significant difference in model performance (95% CIs overlap).')

    if ci_99_llm[1] < ci_99_human[0]:
        print(f"Hypothesis Confirmed (at 99% confidence): LLM-based model performed statistically significantly better (99% CI: {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f} vs {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f}).")
    elif ci_99_human[1] < ci_99_llm[0]:
        print(f"Hypothesis Rejected (at 99% confidence): Human-based model performed statistically significantly better (99% CI: {ci_99_human[0]:.4f}-{ci_99_human[1]:.4f} vs {ci_99_llm[0]:.4f}-{ci_99_llm[1]:.4f}).")
    else:
        print('Result: No statistically significant difference in model performance (99% CIs overlap).')

    print('\nGenerating Visualizations...')
    human_history = {'train': train_rmse_history_human, 'test': test_rmse_history_human}
    llm_history = {'train': train_rmse_history_llm, 'test': test_rmse_history_llm}
    plot_learning_curves(human_history, llm_history)
    plot_error_distribution(model_human, model_llm, test_loader_human, test_loader_llm, device)
    plot_keyword_weights(model_human, model_llm, df, vectorizer_human, vectorizer_llm)

    # RMSE Comparison Plot
    plot_rmse_comparison(
        test_rmse_human, test_rmse_llm,
        ci_95_human, ci_95_llm,
        ci_99_human, ci_99_llm
    )

    # PCA Embedding Visualizations
    visualize_embeddings_pca(model_human, df, vectorizer_human, "Human")
    visualize_embeddings_pca(model_llm, df, vectorizer_llm, "LLM")


if __name__ == '__main__':
    main()
