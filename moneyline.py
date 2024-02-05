from functions import *


if __name__ == "__main__":
    # Loading Games
    df = load_data(load_new_games = False)

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
    #X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=0.20, random_state=70)

    # Train models
    print("Training Models...")
    models, weights = train_models(X_train, Y_train, mlp=True, logit=False, knn=False, rf=False, gb = False)
    print("Models Trained!\n")

    # Creating an ensemble model
    ensemble = False 
    if ensemble:
        model = EnsembleMax(models, weights, 0.501, 0.499)
    else:
        model = models[0]

    evaluate_model(model, X_train, X_test, Y_train, Y_test)

    # Creating a coefficient plot (optional)
    #coefficient_plot(models[0], feature_names)

    # Calibration Plot
    conversion_func, linear_model, poly = calibration_plot(model, X_train, Y_train, X_test, Y_test)

    # Getting the test dataframe for real-time predictions
    test_df = get_test_df(df)

    # Get the games today and odds for them
    teams_to_test = get_todays_odds()
    print(teams_to_test.keys())

    teams_tested = []

    results_df = pd.DataFrame()
    for (team, data) in teams_to_test.items():
        if team not in teams_tested:
            home_team = data['home_team']
            away_team = data['away_team']
            teams_tested.append(home_team)
            teams_tested.append(away_team)
            home_profit = teams_to_test[home_team]["odds"]
            away_profit = teams_to_test[away_team]["odds"]
            temp_df = make_prediction(home_team, away_team, test_df, scaler, model, conversion_func, linear_model, poly, home_profit, away_profit, ensemble = ensemble)
            results_df = pd.concat((results_df, temp_df))
    
    bet_size = 1.00
    results_df["Home Bet"] = 0
    results_df["Away Bet"] = 0
    results_df.loc[results_df["Home LB Return"] > 15, "Home Bet"] = 1
    results_df.loc[results_df["Away LB Return"] > 15, "Away Bet"] = 1
    results_df["Home Bet"] = results_df["Home Bet"]*results_df["Home Prob"] * bet_size
    results_df["Away Bet"] = results_df["Away Bet"]*results_df["Away Prob"] * bet_size

    print(results_df.head())
    results_df.to_excel("data/Results.xlsx", index=False)