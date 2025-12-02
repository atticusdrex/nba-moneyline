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
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import requests, json
from bs4 import BeautifulSoup
from datetime import datetime

def load_data(load_new_games=True, start_date='10/01/1994'):
    print("Loading Games...")
    if load_new_games:
        df = get_games(start_date = start_date)        
        df.to_csv("moneyline/data/RawDF.csv", index=False)
        df = pd.read_csv("moneyline/data/RawDF.csv")
    else:
        df = pd.read_csv("moneyline/data/RawDF.csv")
    
    print("Loaded Games!\n")

    return df

def get_games(start_date = '10/01/1994'):
    # Setup empty dataframe to store the game data 
    games_df = pd.DataFrame()

    # Get a list of teams from the api 
    nba_teams = teams.get_teams()

    # Iterate through all 30 teams and query the data 
    for team in tqdm(nba_teams):
        # Isolate the team ID 
        team_id = team['id']
        # Query for games when this team played from the start_date 
        gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable=start_date,team_id_nullable=team_id, timeout=60)
        # The first DataFrame of those returned is what we want.
        games = gamefinder.get_data_frames()[0]
        
        # concatenate this team's games onto the games_df dataframe 
        games_df = pd.concat((games_df, games))
    
    # Return the df and reset the indices 
    return games_df.reset_index(drop=True)

def get_todays_odds():
    # The Odds API Endpoint for NBA
    # regions=us (gets US books like DraftKings, FanDuel)
    # markets=h2h (Head to Head = Moneyline)
    # oddsFormat=american (e.g. +150, -110)
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": "1d37994e34f2f3f6f2ee4eeffd413052",
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        games_list = response.json()
        full_odds_data = {}
        
        # We will use a counter to create the 0, 1, 2... index keys 
        # to match your previous format
        index_counter = 0

        # Try to load a DraftKings name -> abbreviation mapping if available
        try:
            with open('moneyline/data/DraftkingsNameMatcher.json', 'r') as f:
                name_map = json.load(f)
        except Exception:
            name_map = {}

        # Build a robust lookup of NBA names -> abbreviations using nba_api teams list
        nba_lookup = {}
        try:
            nba_teams = teams.get_teams()
            def _norm(s):
                return ''.join(c for c in str(s).lower() if c.isalnum() or c.isspace()).strip()

            for t in nba_teams:
                abbr = t.get('abbreviation') or t.get('team_abbreviation') or t.get('abbrev')
                full = t.get('full_name') or t.get('team_name') or t.get('nickname')
                nickname = t.get('nickname')
                city = t.get('team_name') or None

                if abbr:
                    # map common forms to abbreviation
                    if full:
                        nba_lookup[_norm(full)] = abbr
                    if nickname:
                        nba_lookup[_norm(nickname)] = abbr
                    if city and nickname:
                        nba_lookup[_norm(f"{city} {nickname}")] = abbr
                    # also include abbreviation itself
                    nba_lookup[_norm(abbr)] = abbr
        except Exception:
            nba_lookup = {}

        import difflib

        def normalize(s: str) -> str:
            return ''.join(c for c in s.lower() if c.isalnum() or c.isspace()).strip()

        def map_name(raw_name: str):
            if raw_name is None:
                return raw_name

            # direct map from provided Draftkings mapping
            if raw_name in name_map:
                return name_map[raw_name]

            norm = normalize(raw_name)

            # direct normalized lookup in DraftKings mapping
            for k, v in name_map.items():
                if normalize(k) == norm:
                    return v

            # try nba_lookup (normalized keys)
            if norm in nba_lookup:
                return nba_lookup[norm]

            # try fuzzy match on combined keys (DraftKings keys + nba_lookup keys)
            keys = list(name_map.keys()) + list(nba_lookup.keys())
            # normalize those keys for matching
            norm_keys = {normalize(k): k for k in keys}
            match = difflib.get_close_matches(raw_name, list(norm_keys.keys()), n=1, cutoff=0.6)
            if match:
                matched_key = norm_keys[match[0]]
                # prefer name_map value if exists else nba_lookup
                if matched_key in name_map:
                    return name_map[matched_key]
                elif match[0] in nba_lookup:
                    return nba_lookup[match[0]]

            # try matching raw against nba_lookup values (abbreviations)
            for v in nba_lookup.values():
                if raw_name.lower() == str(v).lower():
                    return v

            # give up — return original so caller can detect unmapped
            return raw_name

        for game in games_list:
            home_team_name = game['home_team']
            away_team_name = game['away_team']

            # Map DraftKings-style names to team abbreviations used in `test_df`
            home_abbr = map_name(home_team_name)
            away_abbr = map_name(away_team_name)

            # Log unmapped names for visibility
            if home_abbr == home_team_name:
                print(f"Warning: Unmapped home team name from odds API: '{home_team_name}'")
            if away_abbr == away_team_name:
                print(f"Warning: Unmapped away team name from odds API: '{away_team_name}'")
            
            # Find DraftKings odds, or fallback to the first available bookmaker
            # You can change 'draftkings' to 'fanduel', 'betmgm', etc.
            bookmaker = next((bm for bm in game['bookmakers'] if bm['key'] == 'draftkings'), None)
            
            if not bookmaker and game['bookmakers']:
                bookmaker = game['bookmakers'][0] # Fallback
            
            # If no odds exist yet for this game, skip it
            if not bookmaker:
                continue

            # Extract the Moneyline (h2h) market
            market = next((m for m in bookmaker['markets'] if m['key'] == 'h2h'), None)
            if not market:
                continue

            # Get odds for home and away
            # The API returns a list of outcomes, we need to match them to the team names
            outcomes = {o['name']: o['price'] for o in market['outcomes']}
            
            home_odds = outcomes.get(home_team_name)
            away_odds = outcomes.get(away_team_name)

            # Assign to your dictionary format
            # Use the mapped abbreviations as `team` so they match `test_df`'s `TEAM_ABBREVIATION`
            # Entry 1: Away Team (Index 0, 2, 4...)
            full_odds_data[index_counter] = {
                'team': away_abbr,
                'odds': away_odds,
                'home_team': home_abbr,
                'away_team': away_abbr,
                'profit': odds_to_profit(away_odds)
            }
            index_counter += 1

            # Entry 2: Home Team (Index 1, 3, 5...)
            full_odds_data[index_counter] = {
                'team': home_abbr,
                'odds': home_odds,
                'home_team': home_abbr,
                'away_team': away_abbr,
                'profit': odds_to_profit(home_odds)
            }
            index_counter += 1

        return full_odds_data
    
    else:
        # Print the error message from the API if something goes wrong
        raise ValueError(f"Error fetching data: {response.status_code}, {response.text}")

# Function to convert a string of odds data i.e. +250 to an actual proportion 
# profit if the bet is won so 2.5 
def odds_to_profit(odds):
    """
    Converts American odds (e.g. +150, -110) to profit on a $100 bet.
    Handles both string and integer input.
    """
    try:
        # Ensure we have a valid number
        if isinstance(odds, str):
            odds = int(odds.replace("−", "-")) # Handle special minus characters
            
        if odds > 0:
            return (odds / 100)
        else:
            return (100 / abs(odds))
    except (ValueError, TypeError):
        return 0.0


# Making a probability plot to assess the accuracy of the model's probabilistic predictions 
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
    # Filtering out regular season games
    df["SEASON"] = df["SEASON_ID"].astype(str).str[1:].astype(int)
    df["SEASON_TYPE"] = df["SEASON_ID"].astype(str).str[0:1]

    df = df.loc[df["SEASON_TYPE"] == '2',:]
    df = df.drop(columns=["SEASON_TYPE"]).copy()

    # Creating a column to indicate if the home team is playing
    df['Home'] = 1
    # Creating a list of indices to include 
    inds_to_include = []
    # Iterating through the rows
    for (index, row) in tqdm(df.iterrows(), total = len(df)):
        # Setting home games
        if '@' in row.MATCHUP:
            df.loc[index, 'Home'] = 0
        
        # Setting the WL column to be binary instead of W or L
        if row.WL == 'W':
            df.loc[index, 'WL'] = 1
        else:
            df.loc[index, 'WL'] = 0

    # Dropping necessary columns and resetting indices
    df.drop(columns = ['TEAM_NAME', 'MATCHUP', 'SEASON_ID'], inplace=True)
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
    running_totals = df.groupby(['SEASON', 'TEAM_ID']).apply(helper_func).dropna().reset_index(drop=True)
    return running_totals


def groupby_team_season_spread(df):
    def helper_func(group):
        # Sort in ascending order by date
        group['GAME_DATE'] = pd.to_datetime(group['GAME_DATE'])
        group.sort_values(by='GAME_DATE', ascending=True, inplace=True)
        
        # Replace any columns that have NA's with 0 
        group.fillna(0, inplace=True)

        # Creat a "Spread" column for the teams
        group['Spread'] = group['PLUS_MINUS']
        
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
    running_totals = df.groupby(['SEASON', 'TEAM_ID']).apply(helper_func).dropna().reset_index(drop=True)
    return running_totals

def groupby_team_season2(df):
    def helper_func(group):
        game_id = group["GAME_ID"]
        # Isolate the quantitative columns 
        cols = list(group.columns)
        quant_cols = cols[6:]
        # Convert the group with the quantitative columns to numeric
        group[quant_cols] = group[quant_cols].apply(pd.to_numeric)
        # Calculate the game count 
        counts = group["GAME_ID"].expanding(1).count().copy().shift(1)
        running_sum = group[quant_cols].expanding(1).sum().copy().shift(1)
        group.drop(columns = cols[0:5], inplace = True)
        group.reset_index(inplace=True, drop = True)

        for col in quant_cols:
            group[col] = (running_sum[col] / counts).reset_index(drop = True)

        # Calculate win/loss streaks
        streaks = calculate_streaks(group['WL'])
        group['WStreak'], group['LStreak'] = zip(*streaks)
        group[['WStreak', 'LStreak']] = group[['WStreak', 'LStreak']].shift(1)
        group[['WStreak', 'LStreak']].fillna(0, inplace=True)

        # Calculate Home win pct
        group["HomeWinPCT"] = (group["WL"].apply(pd.to_numeric).expanding(1).sum().shift(1).reset_index(drop = True) / counts.reset_index(drop=True))

        # Add the game id 
        group["GAME_ID"] = list(game_id)

        return group

    final_df = df.groupby(by = ["SEASON_ID", "TEAM_ID"]).apply(helper_func).dropna()
    final_df = final_df.reset_index(drop = True)

    return final_df

def match_opponents_optimized(running_totals):
    # Drop unnecessary columns just once
    reduced_totals = running_totals.drop(columns=['SEASON', 'TEAM_ABBREVIATION', 'GAME_DATE'])
    
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

def preprocess_training_spread(match_df, test_size=0.20, random_state = 420):
    X = match_df.drop(columns=['TEAM_ID', 'TEAM_ID_y', 'GAME_ID', 'GAME_ID_y']).dropna().reset_index(drop=True).apply(pd.to_numeric)
    Y = X['Spread'].copy()
    X.drop(columns=['WL', 'Spread'], inplace=True)


    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state = random_state)

    # Scale the input data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Assuming X_train and X_test are your training and testing feature matrices
    poly = PolynomialFeatures(degree=2, include_bias=False)  # Set include_bias=False to exclude the intercept term

    # Generate polynomial features
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    return [X_train, X_test, Y_train, Y_test, list(X.columns), scaler]

def train_models(X_train, Y_train, mlp=True, logit=True, knn=True, rf=True, gb = True):
    models = []
    # Define models 
    if mlp:
        mlp = MLPClassifier((30, 15, 15, 15), activation='tanh',solver='sgd', max_iter=750, warm_start=True, alpha=5e-1, verbose=True, tol=1e-8)
        models.append(mlp)
    if logit:
        logit = LogisticRegressionCV(max_iter=1000)
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

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores}).head(45)

    # Sort features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Create a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.xlabel('Feature Coefficient')
    plt.ylabel('Features')
    plt.title('Stat Contribution to Probability of Team Winning')
    plt.show()


def calibration_plot(model, X_train, Y_train, X_test, Y_test, plot = False):
    from sklearn.linear_model import LinearRegression
    n_bins = 15
    (bounds, probs, samples, errors) = prob_plot(Y_train, model.predict_proba(X_train)[:,1], bins=n_bins)

    x_bounds = 0.5*(bounds[1:]+bounds[0:-1]).copy()
    y_probs = probs.copy()

    poly = PolynomialFeatures(7)
    x_bounds_poly = poly.fit_transform(x_bounds.reshape(-1,1))

    linear_model = LinearRegression()
    linear_model.fit(x_bounds_poly, y_probs)

    def conversion_func(x, linear_model, poly):
        x = poly.transform(x.reshape(-1,1))
        return linear_model.predict(x)

    x_test = np.linspace(0,1,100)
    x_test_poly = poly.transform(x_test.reshape(-1,1))
    y_test = linear_model.predict(x_test_poly)

    if plot:
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

def get_test_df(df, season_year):
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
    test_df = df.groupby(['SEASON', 'TEAM_ID']).apply(helper_func_test).dropna().reset_index(drop=True)
    return test_df[test_df['SEASON'] == season_year].copy()

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
    team2.drop(columns=['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL', 'Home'], inplace=True)
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

    X_test_final = final_df.drop(columns=['SEASON', 'TEAM_ID', 'TEAM_ABBREVIATION', 'GAME_ID', 'GAME_DATE', 'WL']).dropna().reset_index(drop=True).apply(pd.to_numeric)

    # If we have no features (no matching rows found), return an empty-result DataFrame
    if X_test_final.shape[0] == 0:
        df_dict = {
            "Home Team": home_team,
            "Away Team": away_team,
            "Home Odds": int(home_odds) if home_odds is not None else None,
            "Away Odds": int(away_odds) if away_odds is not None else None,
            "Home Prob": np.nan,
            "Away Prob": np.nan,
            "Avg. Home Profit": np.nan,
            "Avg. Away Profit": np.nan,
            "Home Std": np.nan,
            "Away Std": np.nan,
        }

        # Wrap the dict in a list so pandas creates a single-row DataFrame
        return pd.DataFrame([df_dict])

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
        
        "Avg. Home Profit":100*odds_to_profit(home_odds)*(p1-lb_prop) - 100 * (1 - p1+lb_prop),
        "Avg. Away Profit":100 * odds_to_profit(away_odds) * (p2-lb_prop) - 100 * (1 - p1+lb_prop),
        "Home Std": np.sqrt(100**2 * (odds_to_profit(home_odds)+1)**2 * p1*(1-p1)),
        "Away Std": np.sqrt(100**2 * (odds_to_profit(away_odds)+1)**2 * p2*(1-p2)),
    }

    return(pd.DataFrame(df_dict))
    
    

# Kernel function
def k(x, y, s):
    v = x - y
    return np.exp(-np.inner(v,v) / (s))


# Function to create kernel matrix 
def K(X_left, X_right, s):
    mat = np.zeros((X_left.shape[1], X_right.shape[1]))
    for i in tqdm(range(mat.shape[0])):
        for j in range(mat.shape[1]):
            mat[i,j] = k(X_left[:,i], X_right[:,j], s)
    
    return mat