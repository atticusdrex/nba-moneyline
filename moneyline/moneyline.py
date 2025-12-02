from .util import *

# FOR A DRY RUN FOR WAGERING: 
# Load new games -> True 
# Set the test_size to smallest possible setting 

if __name__ == "__main__":
    # Loading Games
    df = load_data(load_new_games = True)

    print("Processing Dataset...")
    # Preprocessing dataset
    df = filter_dataset(df)

    # Grouping the data by team & season
    running_totals = groupby_team_season(df)

    # Matching teams to their opponents
    match_df = match_opponents_optimized(running_totals)

    print("Preprocessing Finished!\n")

    # Split into training and validation sets
    X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=2/len(match_df), random_state=420)
    #X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=0.20, random_state=420)

    # Train models
    print("Training Models...")
    models, weights = train_models(X_train, Y_train, mlp=False, logit=True, knn=False, rf=False, gb = False)
    print("Models Trained!\n")

    # Creating an ensemble model
    ensemble = False 
    if ensemble:
        model = EnsembleMax(models, weights, 0.501, 0.499)
    else:
        model = models[0]

    evaluate_model(model, X_train, X_test, Y_train, Y_test)

    # Creating a coefficient plot (optional)
    # coefficient_plot(models[0], feature_names)

    # Calibration Plot
    conversion_func, linear_model, poly = calibration_plot(model, X_train, Y_train, X_test, Y_test, plot = False)

    # Getting the test dataframe for real-time predictions
    test_df = get_test_df(df, 2025)

    # Get the games today and odds for them
    teams_to_test = get_todays_odds()
    teams_tested = []

    results_df = pd.DataFrame()
    for (i, data) in teams_to_test.items():
        team = data['team']
        if (i+1)%2==0:
            home_data = data 
            away_data = teams_to_test[i-1]

            home_team = home_data['team']
            away_team = away_data['team']

            home_profit = home_data['odds']
            away_profit = away_data['odds']

            temp_df = make_prediction(home_team, away_team, test_df, scaler, model, conversion_func, linear_model, poly, home_profit, away_profit, ensemble = ensemble)
            results_df = pd.concat((results_df, temp_df))
            

    print(results_df.head())
    results_df.to_csv("moneyline/data/results.csv", index=False)