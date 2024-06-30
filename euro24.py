import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# dataframe for upcoming matches
upcoming_matches = pd.DataFrame({
    'date': [
        '2024-06-14', '2024-06-15', '2024-06-15', '2024-06-15',
        '2024-06-16', '2024-06-16', '2024-06-16', '2024-06-17',
        '2024-06-17', '2024-06-17', '2024-06-18', '2024-06-18',
        '2024-06-19', '2024-06-19', '2024-06-19', '2024-06-20',
        '2024-06-20', '2024-06-20', '2024-06-21', '2024-06-21',
        '2024-06-21', '2024-06-22', '2024-06-22', '2024-06-22',
        '2024-06-23', '2024-06-23', '2024-06-24', '2024-06-24',
        '2024-06-25', '2024-06-25', '2024-06-25', '2024-06-25',
        '2024-06-26', '2024-06-26', '2024-06-26', '2024-06-26'
    ],
    'home_team': [
        'Germany', 'Hungary', 'Spain', 'Italy', 'Slovenia', 'Serbia', 'Poland', 'Austria',
        'Romania', 'Belgium', 'Turkey', 'Portugal', 'Germany', 'Scotland', 'Croatia', 'Spain',
        'Slovenia', 'Denmark', 'Poland', 'Netherlands', 'Slovakia', 'Belgium', 'Georgia', 'Turkey',
        'Germany', 'Scotland', 'Albania', 'Croatia', 'England', 'Denmark', 'Netherlands', 'France',
        'Slovakia', 'Ukraine', 'Georgia', 'Czech Republic'
    ],
    'away_team': [
        'Scotland', 'Switzerland', 'Croatia', 'Albania', 'Denmark', 'England', 'Netherlands', 'France',
        'Ukraine', 'Slovakia', 'Georgia', 'Czech Republic', 'Hungary', 'Switzerland', 'Albania', 'Italy',
        'Serbia', 'England', 'Austria', 'France', 'Ukraine', 'Romania', 'Czech Republic', 'Portugal',
        'Switzerland', 'Hungary', 'Spain', 'Italy', 'Slovenia', 'Serbia', 'Austria', 'Poland',
        'Romania', 'Belgium', 'Portugal', 'Turkey'
    ],
    'neutral': [False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True,
                True, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True,
                True, True, True, True]
})

upcoming_matches['date'] = pd.to_datetime(upcoming_matches['date'])

def calculate_rolling_averages(matches):
    matches = matches.sort_values(by='date')

    # Rolling averages for home team
    home_rolling_avg = matches.groupby('home_team').rolling(3, on='date').agg({
        'home_score': 'mean',
        'away_score': 'mean'
    }).reset_index()
    home_rolling_avg = home_rolling_avg.rename(columns={'home_score': 'home_rolling_avg_goals_scored', 'away_score': 'home_rolling_avg_goals_conceded'})
    home_rolling_avg = home_rolling_avg[['home_team', 'date', 'home_rolling_avg_goals_scored', 'home_rolling_avg_goals_conceded']]

    # Merge rolling averages back to matches DataFrame
    matches = matches.merge(home_rolling_avg, on=['home_team', 'date'], how='left')

    return matches

def get_most_recent_non_nan(group):

    group = group.dropna(subset=['home_rolling_avg_goals_scored', 'home_rolling_avg_goals_conceded'])
    group = group.sort_values(by='date', ascending=False)
    if not group.empty:
        return group.iloc[0]
    else:
        return pd.Series([np.nan] * len(group.columns), index=group.columns)

def prepare_input_data(upcoming_matches, historical_data, fifa_rankings_df):
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    upcoming_matches['date'] = pd.to_datetime(upcoming_matches['date'])

    # rolling avg using historical data
    historical_data_with_rolling_avg = calculate_rolling_averages(historical_data)

    # most recent non-NaN rolling averages for home teams in upcoming matches
    most_recent_rolling_avg = historical_data_with_rolling_avg.groupby('home_team').apply(get_most_recent_non_nan).reset_index(drop=True)

    # FIFA rankings
    upcoming_matches = upcoming_matches.merge(fifa_rankings_df, left_on='home_team', right_on='team', how='left')
    upcoming_matches = upcoming_matches.rename(columns={'ranking': 'home_team_ranking', 'points': 'home_team_points'}).drop('team', axis=1)
    upcoming_matches = upcoming_matches.merge(fifa_rankings_df, left_on='away_team', right_on='team', how='left')
    upcoming_matches = upcoming_matches.rename(columns={'ranking': 'away_team_ranking', 'points': 'away_team_points'}).drop('team', axis=1)

    upcoming_matches = upcoming_matches.merge(most_recent_rolling_avg[['home_team', 'home_rolling_avg_goals_scored', 'home_rolling_avg_goals_conceded']],
                                              on='home_team', how='left')

    # rolling averages with 0 (or other appropriate values)
    upcoming_matches['home_rolling_avg_goals_scored'] = upcoming_matches['home_rolling_avg_goals_scored'].fillna(0)
    upcoming_matches['home_rolling_avg_goals_conceded'] = upcoming_matches['home_rolling_avg_goals_conceded'].fillna(0)

    # Encode teams
    upcoming_matches['opponent_code'] = upcoming_matches['away_team'].astype("category").cat.codes
    upcoming_matches['team_code'] = upcoming_matches['home_team'].astype("category").cat.codes

    # Order columns as specified
    cols_order = ['neutral', 'home_rolling_avg_goals_scored', 'home_rolling_avg_goals_conceded',
                  'home_team_ranking', 'home_team_points', 'away_team_ranking', 'away_team_points',
                  'opponent_code', 'team_code']
    upcoming_matches = upcoming_matches[cols_order]

    return upcoming_matches

data = pd.read_csv("/Users/danteamicarella/Downloads/archive 2/results.csv")
fifa_rankings_df = pd.read_csv('/Users/danteamicarella/fifa_rankings.csv')

X_upcoming = prepare_input_data(upcoming_matches, data, fifa_rankings_df)
print(X_upcoming)

# Predict Outcomes
def predict_outcomes(model, X_upcoming, model_name):
    predictions = model.predict(X_upcoming)
    print(f'{model_name} Predictions for Upcoming Matches:')
    print(predictions)
    return predictions

def preprocess_data(data):
    fifa_rankings_df = pd.read_csv('/Users/danteamicarella/fifa_rankings.csv')

    data['date'] = pd.to_datetime(data['date'])
    matches = data[~(data['date'] < '2018-01-01')]

    matches = calculate_rolling_averages(matches)

    matches = matches.merge(fifa_rankings_df, left_on='home_team', right_on='team', how='left')
    matches = matches.rename(columns={'ranking': 'home_team_ranking', 'points': 'home_team_points'}).drop('team', axis=1)
    matches = matches.merge(fifa_rankings_df, left_on='away_team', right_on='team', how='left')
    matches = matches.rename(columns={'ranking': 'away_team_ranking', 'points': 'away_team_points'}).drop('team', axis=1)

    matches['home_rolling_avg_goals_scored'] = matches['home_rolling_avg_goals_scored'].fillna(0)
    matches['home_rolling_avg_goals_conceded'] = matches['home_rolling_avg_goals_conceded'].fillna(0)
    matches = matches.dropna(subset=['home_team_ranking', 'away_team_ranking', 'home_team_points', 'away_team_points'])
    matches = matches.copy()

    matches['opponent_code'] = matches['away_team'].astype("category").cat.codes
    matches['team_code'] = matches['home_team'].astype("category").cat.codes

    # target variable: 2 for home win, 1 for draw, 0 for away win
    matches['result'] = matches.apply(lambda row: 2 if row['home_score'] > row['away_score'] else (1 if row['home_score'] == row['away_score'] else 0), axis=1)

    # Drop unnecessary columns
    matches = matches.drop(columns=['date', 'home_score', 'away_score', 'tournament', 'city', 'country', 'home_team', 'away_team'])

    X = matches.drop(columns=['result'])
    y = matches['result']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=['Away Win', 'Draw', 'Home Win'], columns=['Predicted Away Win', 'Predicted Draw', 'Predicted Home Win'])
    print(f"{model_name} Confusion Matrix:")
    print(conf_matrix_df)

    precision = precision_score(y_test, y_pred, average=None)
    print(f'{model_name} Precision:')
    for i, p in enumerate(precision):
        print(f'Class {i}: {p}')

X_train, X_test, y_train, y_test = preprocess_data(data)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train and evaluate models
# RandomForest Model
best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    random_state=42
)
best_rf.fit(X_train_balanced, y_train_balanced)
evaluate_model(best_rf, X_test, y_test, "Random Forest")

# SVM Model
svm_model = SVC(kernel='poly', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train_balanced, y_train_balanced)
evaluate_model(svm_model, X_test, y_test, "SVM")

# Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(150, 100), activation='relu', solver='adam', max_iter=500, random_state=42)
nn_model.fit(X_train_balanced, y_train_balanced)
evaluate_model(nn_model, X_test, y_test, "Neural Network")

# Predict outcomes for upcoming matches
rf_predictions = predict_outcomes(best_rf, X_upcoming, "Random Forest")
svm_predictions = predict_outcomes(svm_model, X_upcoming, "SVM")
nn_predictions = predict_outcomes(nn_model, X_upcoming, "Neural Network")


upcoming_matches['RF_Prediction'] = rf_predictions
upcoming_matches['SVM_Prediction'] = svm_predictions
upcoming_matches['NN_Prediction'] = nn_predictions

print(upcoming_matches)

from matplotlib.colors import ListedColormap

groups = {
    'Group A': ['Germany', 'Scotland', 'Hungary', 'Switzerland'],
    'Group B': ['Spain', 'Croatia', 'Italy', 'Albania'],
    'Group C': ['Slovenia', 'Denmark', 'Serbia', 'England'],
    'Group D': ['Netherlands', 'France', 'Poland', 'Austria'],
    'Group E': ['Ukraine', 'Slovakia', 'Belgium', 'Romania'],
    'Group F': ['Portugal', 'Czech Republic', 'Georgia', 'Turkey']
}

def plot_agreement_heatmap(df, title):
    agreement_matrix = df[['RF_Prediction', 'SVM_Prediction', 'NN_Prediction']]

    prediction_cmap = ListedColormap(["#b3bfd1", "#a4a2a8", "#df8879"])  # Colors: Away Win, Draw, Home Win

    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create the heatmap for predictions
    sns.heatmap(agreement_matrix.T, cmap=prediction_cmap, cbar=False, annot=True, fmt='d', linewidths=.5, annot_kws={"size": 10}, ax=ax)
    
    plt.title(f'{title} - Model Agreement on Match Predictions', fontsize=18)
    plt.xlabel('Matchup', fontsize=14)
    plt.ylabel('Model', fontsize=14)
    
    # Create matchup labels for x-axis
    matchups = ["{} vs. {}".format(row['home_team'], row['away_team']) for _, row in df.iterrows()]
    ax.set_xticks([i + 0.5 for i in range(len(matchups))])
    ax.set_xticklabels(matchups, rotation=45, fontsize=10)
    
    plt.yticks(fontsize=12, rotation=0)
    
    legend_labels = {
        0: 'Away Win',
        1: 'Draw',
        2: 'Home Win'
    }
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#b3bfd1', lw=4),
                    Line2D([0], [0], color='#a4a2a8', lw=4),
                    Line2D([0], [0], color='#df8879', lw=4)]
    
    plt.legend(custom_lines, [legend_labels[key] for key in legend_labels.keys()], title="Predictions",
               bbox_to_anchor=(-0.2, 1), loc='upper left', borderaxespad=0.)
    
    plt.show()

# Function to filter matches by group
def filter_matches_by_group(df, group_teams):
    return df[df['home_team'].isin(group_teams) & df['away_team'].isin(group_teams)]

# Iterate over each group and plot the heatmap
for group_name, teams in groups.items():
    group_matches = filter_matches_by_group(upcoming_matches, teams)
    plot_agreement_heatmap(group_matches, group_name)