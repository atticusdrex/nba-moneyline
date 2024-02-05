import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
import pytz 
from datetime import datetime
import requests, json
from bs4 import BeautifulSoup

def load_data(load_new_games=True, start_date='10/01/1994'):
    print("Loading Games...")
    if load_new_games:
        df = get_games(start_date = start_date)        
        df.to_csv("data/RawDF.csv", index=False)
        df = pd.read_csv("data/RawDF.csv")
    else:
        df = pd.read_csv("data/RawDF.csv")
    
    print("Loaded Games!\n")

    return df

def get_games(start_date = '10/01/1994'):
    games_df = pd.DataFrame()

    nba_teams = teams.get_teams()

    for team in tqdm(nba_teams):
        team_id = team['id']
        # Query for games where the Celtics were playing
        gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable=start_date,team_id_nullable=team_id, timeout=60)
        # The first DataFrame of those returned is what we want.
        games = gamefinder.get_data_frames()[0]
        
        games_df = pd.concat((games_df, games))
    return games_df.reset_index(drop=True)

def get_todays_odds():
    url = "https://sportsbook.draftkings.com/leagues/basketball/nba"

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        tbody = soup.find('tbody')
        odds_data = {}

        with open("data/DraftkingsNameMatcher.json","r") as infile:
            names_matcher = json.load(infile)

        # Assuming each row is within a <tr> tag and each cell within a <td> tag
        for (name, data) in zip(tbody.find_all('th'), tbody.find_all('tr')):
            # Extract the text from each cell in the row and add to the row_data list
            name = [cell.text.strip() for cell in name.find_all('div')][-1]
            
            name = names_matcher[name]
            
            moneyline_odds = [cell.text.strip() for cell in data.find_all('td')][-1]
            odds_data[name] = moneyline_odds.replace("−","-")
            # Seeing if it's a home game
        
        full_odds_data = {}
        for (i, (team, odds)) in enumerate(odds_data.items()):
            if (i+1)%2 == 0:
                full_odds_data[team] = {
                    "odds":odds,
                    "home_team":team,
                    "away_team":list(odds_data.keys())[i-1],
                    "profit":odds_to_profit(odds)
                }
            else:
                full_odds_data[team] = {
                    "odds":odds,
                    "home_team":list(odds_data.keys())[i+1],
                    "away_team":team,
                    "profit":odds_to_profit(odds)
                }
        
        return full_odds_data
    else:
        raise ValueError("No Response Received from Draftkings!")


def odds_to_profit(odds):
    if odds[0] == "+":
        odds = float(odds[1:])
        return odds / 100
    else:
        odds = float(odds[1:])
        return 100 / odds


    
        
def prob_plot(Y_true, Y_pred, bins=25):
    probs = np.zeros((bins))
    samples = np.zeros_like(probs)
    errors = np.zeros_like(probs)
    bound_vec = np.linspace(0, 1, bins + 1)
    bounds = zip(bound_vec[0:-1], bound_vec[1:])
    
    for (i, (lower, upper)) in enumerate(bounds):
        total = len((Y_pred[(Y_pred >= lower) & (Y_pred < upper)]))
        correct = np.sum(Y_true[(Y_pred >= lower) & (Y_pred < upper)])
        if total > 0:
            probs[i] = correct / total
            samples[i] = total / len(Y_pred)
            errors[i] = 2.5 * np.sqrt(probs[i]*(1 - probs[i]) / total)
        else:
            probs[i] = 0.5*(upper + lower)
            samples[i] = total / len(Y_pred)
            errors[i] = 0.5
    
    return (bound_vec, probs, samples, errors)


def filter_dataset(df):
    # Creating a column to indicate if the home team is playing
    df['Home'] = 1
    # Creating a list of indices to include 
    inds_to_include = []
    # Iterating through the rows
    for (index, row) in tqdm(df.iterrows()):
        # Getting rid of the first character on the SEASON_ID
        row.SEASON_ID = str(row.SEASON_ID)[1:]
        df.loc[index, 'SEASON_ID'] = row.SEASON_ID
        # Setting home games
        if '@' in row.MATCHUP:
            df.loc[index, 'Home'] = 0
        
        # Setting the WL column to be binary instead of W or L
        if row.WL == 'W':
            df.loc[index, 'WL'] = 1
        else:
            df.loc[index, 'WL'] = 0

        # Only including the in-season months
        month = int(row.GAME_DATE.split('-')[1])
        day = int(row.GAME_DATE.split('-')[2])
        if month < 7 or month > 9:
            if month == 10 and day > 20:
                inds_to_include.append(index)
            elif month > 10 or month < 7:
                inds_to_include.append(index)

    # Dropping necessary columns and resetting indices
    df = df.loc[inds_to_include, :]
    df.drop(columns = ['TEAM_NAME', 'MATCHUP'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

quant_cols = ['MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

def calculate_streaks(series):
        # Initialize streak counters
        WStreak, LStreak = 0, 0
        streaks = []
        for result in series:
            if result == 1:
                WStreak += 1
                LStreak = 0
            elif result == 0:
                LStreak += 1
                WStreak = 0
            else:
                WStreak, LStreak = 0, 0  # Reset streaks for non-W/L results
            streaks.append((WStreak, LStreak))
        return streaks

def groupby_team_season(df):

    def helper_func(group):
        # Sort in ascending order by date
        group['GAME_DATE'] = pd.to_datetime(group['GAME_DATE'])
        group.sort_values(by='GAME_DATE', ascending=True, inplace=True)
        
        # Replace any columns that have NA's with 0 
        group.fillna(0, inplace=True)
        
        # Create running averages for quantitative columns
        group[quant_cols] = group[quant_cols].expanding(1).sum().shift(1).copy()
        group['Count'] = group['GAME_DATE'].expanding(1).count().shift(1).copy()
        for col in quant_cols:
            group[col] = group[col] / group['Count']
        
        # Convert WL column into a win percentage
        group['WIN_PCT'] = group['WL'].expanding(1).sum().shift(1).copy() / group['Count']
        
        # Calculate win/loss streaks
        streaks = calculate_streaks(group['WL'])
        group['WStreak'], group['LStreak'] = zip(*streaks)
        group[['WStreak', 'LStreak']] = group[['WStreak', 'LStreak']].shift(1)
        group[['WStreak', 'LStreak']].fillna(0, inplace=True)

        # Compute Home Win Percentage
        home_games_mask = group['Home'] == 1
        home_wins = group['WL'][home_games_mask].expanding().apply(lambda x: (x == 1).sum())
        total_home_games = home_games_mask.expanding().sum()
        group['HomeWinPct'] = home_wins / total_home_games
        group['HomeWinPct'].fillna(method='ffill', inplace=True)  # Set away game values correctly
        group['HomeWinPct'] = group['HomeWinPct'].shift(1).fillna(0)

        # Remove the Count column
        group.drop(columns='Count', inplace=True)

        return group

    # Sort by unique SEASON_ID and TEAM_ID, apply the helper_func(), drop all the NA's and reset the indices
    running_totals = df.groupby(['SEASON_ID', 'TEAM_ID']).apply(helper_func).dropna().reset_index(drop=True)
    return running_totals

def match_opponents(running_totals):
    # Create a dataframe for the matchups
    match_df = pd.DataFrame()

    for (index, row) in tqdm(running_totals.iterrows()):
        # Get the current team's id 
        team_id = row.TEAM_ID
        # Find the opponent's stats
        other_team = running_totals[(running_totals['GAME_ID'] == row.GAME_ID) & (running_totals['TEAM_ID'] != row.TEAM_ID)]
        # Drop the unnecessary/redundant columns
        other_team = other_team.drop(columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL', 'Home'])
        # Reset the index
        other_team.reset_index(inplace=True, drop=True)
        # Create new columns with suffix _y for the other team
        new_cols = [col + "_y" for col in list(other_team.columns)]
        # Save new columns as the other team's columns
        other_team.columns = new_cols
        # Combine this team and other team data
        this_row = pd.DataFrame([row]).reset_index(drop=True)
        this_row = pd.concat((this_row, other_team), axis=1)
        #for col in list(other_team.columns):
            #this_row[col] = other_team[col]
        # Concatenate the new dataframe with this one 
        match_df = pd.concat((match_df, this_row))

    return match_df

def match_opponents_optimized(running_totals):
    # Drop unnecessary columns just once
    reduced_totals = running_totals.drop(columns=['SEASON_ID', 'TEAM_ABBREVIATION', 'GAME_DATE'])
    
    # Create two DataFrames, one for each team in a game, and shift column names for the second team
    team_1 = reduced_totals.copy()
    team_2 = reduced_totals.add_suffix('_y').drop(columns=['WL_y', 'Home_y'])

    # Merge these DataFrames based on the GAME_ID, ensuring different teams are matched
    merged_df = pd.merge(team_1, team_2, left_on=['GAME_ID'], right_on=['GAME_ID_y'])
    
    # Filter out rows where the team IDs are the same, as we only want matchups
    match_df = merged_df[merged_df['TEAM_ID'] != merged_df['TEAM_ID_y']]

    return match_df

def preprocess_training(match_df, test_size=0.20, random_state = 420):
    X = match_df.drop(columns=['TEAM_ID', 'TEAM_ID_y', 'GAME_ID', 'GAME_ID_y']).dropna().reset_index(drop=True).apply(pd.to_numeric)
    Y = X['WL'].copy()
    X.drop(columns=['WL'], inplace=True)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)

    # Scale the input data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return [X_train, X_test, Y_train, Y_test, list(X.columns), scaler]

def train_models(X_train, Y_train, mlp=True, logit=True, knn=True, rf=True, gb = True):
    models = []
    # Define models 
    if mlp:
        mlp = MLPClassifier((30, 15, 15, 15), activation='tanh',solver='sgd', max_iter=750, warm_start=True, alpha=5e-1, verbose=True, tol=1e-8)
        models.append(mlp)
    if logit:
        logit = LogisticRegressionCV()
        models.append(logit)
    if knn:
        knn = KNeighborsClassifier(n_neighbors=150)
        models.append(knn)
    if rf:
        rf = RandomForestClassifier(max_depth=5, n_estimators=100, n_jobs=-1)
        models.append(rf)
    if gb:
        gb = GradientBoostingClassifier()
        models.append(gb)



    # Fit each models to the training data
    for (i, model) in tqdm(enumerate(models)):
        #model = CalibratedClassifierCV(model, method = "isotonic")
        model.fit(X_train, Y_train)
        models[i] = model

    # Assign weights based on ROC-AUC Score or Accuracy
    weights = []
    [weights.append(roc_auc_score(Y_train, model.predict(X_train))) for model in models]

    # Normalize the weights
    weights = [weight / sum(weights) for weight in weights]

    return [models, weights]

class EnsembleMax:
    def __init__(self, models, weights, top_thresh, bottom_thresh):
        self.models = models
        self.weights = weights 
        self.upper = top_thresh 
        self.lower = bottom_thresh
    
    def predict(self, X):
        preds = np.zeros((X.shape[0], len(self.models)))
        for j in range(len(self.models)):
            preds[:,j] = self.models[j].predict_proba(X)[:,1]
        
        means = np.mean(preds, axis=1)
        maxs = np.max(preds, axis=1)
        mins = np.min(preds, axis=1)

        final_preds = -np.ones_like(means)
        final_preds[means > self.upper] = maxs[means > self.upper]
        final_preds[means <= self.lower] = mins[means < self.lower]
        final_preds[final_preds == -1] = means[final_preds == -1]
        
        final_preds = np.round(final_preds)
        return final_preds
        #return np.round(np.array(sum([weight * model.predict(X) for (weight, model) in zip(self.weights, self.models)])))

    def predict_proba(self, X):
        preds = np.zeros((X.shape[0], len(self.models)))
        for j in range(len(self.models)):
            preds[:,j] = self.models[j].predict_proba(X)[:,1]
        
        means = np.mean(preds, axis=1)
        maxs = np.max(preds, axis=1)
        mins = np.min(preds, axis=1)

        final_preds = -np.ones_like(means)
        final_preds[means > self.upper] = maxs[means > self.upper]
        final_preds[means < self.lower] = mins[means < self.lower]
        final_preds[final_preds == -1] = means[final_preds == -1]
        
        final_preds = np.reshape(final_preds, (len(final_preds), 1))
        return np.concatenate(((1-final_preds), final_preds), axis=1)
        #return np.array(sum([weight * model.predict_proba(X) for (weight, model) in zip(self.weights, self.models)]))
    
class Ensemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights 
    
    def predict(self, X):
        return np.round(np.array(sum([weight * model.predict(X) for (weight, model) in zip(self.weights, self.models)])))

    def predict_proba(self, X):
        return np.array(sum([weight * model.predict_proba(X) for (weight, model) in zip(self.weights, self.models)]))
    
    def predict_CI(self, X, axis=1):
        preds = np.zeros((X.shape[0], len(self.models)))
        for j in range(len(self.models)):
            preds[:,j] = self.models[j].predict_proba(X)[:,axis]
        
        means = np.mean(preds, axis=1)
        maxs = np.max(preds, axis=1)
        mins = np.min(preds, axis=1)

        return maxs, mins, means

    
def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    # Print the statistics
    print("Train Accuracy: %.3f %%" % (100 * accuracy_score(Y_train, model.predict(X_train))))
    print("Test  Accuracy: %.3f %%" % (100 * accuracy_score(Y_test, model.predict(X_test))))
    print(confusion_matrix(Y_test, model.predict(X_test)))
    print("Train ROC Score: %.4f" % (roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])))
    print("Test ROC Score: %.4f" % (roc_auc_score(Y_test, model.predict_proba(X_test)[:,1]))) if len(Y_test) > 50 else print("")

def coefficient_plot(logit, feature_names ):
    importance_scores = logit.coef_[0]

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores}).head(30)

    # Sort features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.xlabel('Feature Coefficient')
    plt.ylabel('Features')
    plt.title('Stat Contribution to Probability of Team Winning')
    plt.show()


def calibration_plot(model, X_train, Y_train, X_test, Y_test):
    from sklearn.linear_model import LinearRegression
    n_bins = 18
    (bounds, probs, samples, errors) = prob_plot(Y_train, model.predict_proba(X_train)[:,1], bins=n_bins)

    x_bounds = 0.5*(bounds[1:]+bounds[0:-1]).copy()
    y_probs = probs.copy()

    poly = PolynomialFeatures(9)
    x_bounds_poly = poly.fit_transform(x_bounds.reshape(-1,1))

    linear_model = LinearRegression()
    linear_model.fit(x_bounds_poly, y_probs)

    def conversion_func(x, linear_model, poly):
        x = poly.transform(x.reshape(-1,1))
        return linear_model.predict(x)

    x_test = np.linspace(0,1,100)
    x_test_poly = poly.transform(x_test.reshape(-1,1))
    y_test = linear_model.predict(x_test_poly)

    plt.figure(figsize=(10,5))
    plt.scatter(x_bounds, y_probs, label = "Raw Probabilities")
    plt.plot(x_test, y_test, label = "Polynomial Approximation")
    plt.scatter(0.5*(bounds[1:]+bounds[0:-1]), samples / np.max(samples), label = "Sample Size Proportions")
    plt.errorbar(0.5*(bounds[1:]+bounds[0:-1]), probs, yerr=errors, ecolor='black', capsize=3)
    plt.plot([0,1], [0,1], label = "Ideal Relationship")
    plt.grid()
    plt.legend()
    plt.xlabel("Probability Predicted Won")
    plt.ylabel("Actual Probability Won")
    plt.title("Calibration Curve with 98.75% Confidence Intervals")

    plt.show()

    plt.figure(figsize=(10,5))
    (bounds, probs, samples, errors) = prob_plot(Y_test, conversion_func(model.predict_proba(X_test)[:,1], linear_model, poly), bins=n_bins)
    plt.scatter(x_bounds, probs, c='red', label = "Corrected Points")
    plt.errorbar(x_bounds, probs, yerr=errors, ecolor='black', capsize=3)
    (bounds, probs, samples, errors) = prob_plot(Y_test, model.predict_proba(X_test)[:,1], bins=n_bins)
    plt.scatter(x_bounds, probs, c='green', label="Uncorrected Points")
    plt.errorbar(x_bounds, probs, yerr=errors, ecolor='blue', capsize=3)
    plt.plot([0,1], [0,1])
    plt.grid()
    plt.xlabel("Probability Predicted Won")
    plt.ylabel("Actual Probability Won")
    plt.title("Calibration Curve with Calibrated Probabilities (98.75 Confidence Intervals)")
    plt.legend()

    plt.show()

    return [conversion_func, linear_model, poly]

def get_test_df(df):
    def helper_func_test(group):
        # Sort in ascending order by date
        group['GAME_DATE'] = pd.to_datetime(group['GAME_DATE'])
        group.sort_values(by='GAME_DATE', ascending=True, inplace=True)
        
        # Replace any columns that have NA's with 0 
        group.fillna(0, inplace=True)
        
        # Create running averages for quantitative columns
        group[quant_cols] = group[quant_cols].expanding(1).sum().copy()
        group['Count'] = group['GAME_DATE'].expanding(1).count().copy()
        for col in quant_cols:
            group[col] = group[col] / group['Count']
        
        # Convert WL column into a win percentage
        group['WIN_PCT'] = group['WL'].expanding(1).sum().copy() / group['Count']
        
        # Calculate win/loss streaks
        streaks = calculate_streaks(group['WL'])
        group['WStreak'], group['LStreak'] = zip(*streaks)
        #group[['WStreak', 'LStreak']] = group[['WStreak', 'LStreak']].shift(1)
        #group[['WStreak', 'LStreak']].fillna(0, inplace=True)

        # Compute Home Win Percentage
        home_games_mask = group['Home'] == 1
        home_wins = group['WL'][home_games_mask].expanding().apply(lambda x: (x == 1).sum())
        total_home_games = home_games_mask.expanding().sum()
        group['HomeWinPct'] = home_wins / total_home_games
        group['HomeWinPct'].fillna(method='ffill', inplace=True)  # Set away game values correctly
        #group['HomeWinPct'] = group['HomeWinPct'].shift(1).fillna(0)

        # Remove the Count column
        group.drop(columns='Count', inplace=True)

        return group

    # Sort by unique SEASON_ID and TEAM_ID, apply the helper_func(), drop all the NA's and reset the indices
    test_df = df.groupby(['SEASON_ID', 'TEAM_ID']).apply(helper_func_test).dropna().reset_index(drop=True)
    return test_df[test_df['SEASON_ID'] == '2023'].copy()

def make_prediction(home_team, away_team, test_df, scaler, model, conversion_func, linear_model, poly, home_odds, away_odds, ensemble=False):
    team1 = test_df[test_df['TEAM_ABBREVIATION'] == home_team].tail(1).copy() # Home Team
    team2 = test_df[test_df['TEAM_ABBREVIATION'] == away_team].tail(1).copy()

    for col in team1.columns:
        if '_y' in col:
            team1.drop(columns = [col], inplace=True)
            team2.drop(columns = [col], inplace=True)

    # Set team1 as the home team 
    team1["Home"] = 1
    pd.concat((team1, team2)).head()

    final_df = pd.DataFrame()
    # Drop the unnecessary/redundant columns
    team2.drop(columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL', 'Home'], inplace=True)
    # Reset the index
    team1.reset_index(inplace=True, drop=True)
    team2.reset_index(inplace=True, drop=True)
    # Create new columns with suffix _y for the other team
    new_cols = [col + "_y" for col in list(team2.columns)]
    # Save new columns as the other team's columns
    team2.columns = new_cols
    # Combine this team and other team data
    for col in list(team2.columns):
        team1[col] = team2[col]
    # Concatenate the new dataframe with this one 
    final_df = pd.concat((final_df, team1))

    X_test_final = final_df.drop(columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL']).dropna().reset_index(drop=True).apply(pd.to_numeric)
    X_test_final = scaler.transform(X_test_final)

    #print(away_team + " @ " + home_team)

    p1 = conversion_func(model.predict_proba(X_test_final)[0][1], linear_model, poly)
    p2 = conversion_func(model.predict_proba(X_test_final)[0][0], linear_model, poly)
    lb_prop = 0.05

    df_dict = {
        "Home Team":home_team,
        "Away Team":away_team,
        "Home Odds":int(home_odds),
        "Away Odds":int(away_odds),
        "Home Prob":p1,
        "Away Prob":p2,
        "Home LB Return":100*odds_to_profit(home_odds)*(p1-lb_prop) - 100 * (1 - p1+lb_prop),
        "Away LB Return":100 * odds_to_profit(away_odds) * (p2-lb_prop) - 100 * (1 - p1+lb_prop)
    }

    return(pd.DataFrame(df_dict))

    
    print("Probability that %s wins: %.2f%%, Exp. Profit: %.3f%% (LB: %.3f%%)" % (home_team, 100 * p1, 100*home_odds*p1 - 100 * (1 - p1), 100*home_odds*(p1-lb_prop) - 100 * (1 - p1+lb_prop)))
    print("Probability that %s wins: %.2f%%, Exp. Profit: %.3f%% (LB: %.3f%%)\n" % (away_team, 100 * p2, 100 * away_odds * p2 - 100 * (1 - p1), 100 * away_odds * (p2-lb_prop) - 100 * (1 - p1+lb_prop)))
    
    

