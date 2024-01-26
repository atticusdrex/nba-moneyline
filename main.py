from functions import *

load_new_games = True

if __name__ == "__main__":
    # Loading Games
    if load_new_games:
        df = get_games()        
        df.to_csv("RawDF.csv", index=False)
        df = pd.read_csv("RawDF.csv")
    else:
        df = pd.read_csv("RawDF.csv")
    
    print("Loaded Games!")

    # Preprocessing dataset
    df = filter_dataset(df)

    # Grouping the data by team & season
    running_totals = groupby_team_season(df)

    # Matching teams to their opponents
    match_df = match_opponents_optimized(running_totals)

    print("Preprocessing Finished!")

    # Split into training and validation sets
    #X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=2/len(match_df), random_state=420)
    X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=0.20, random_state=420)

    # Train models
    models, weights = train_models(X_train, Y_train)
    print("Models Trained!")

    # Creating an ensemble model
    model = EnsembleMax(models, weights, 0.501, 0.499)
    #model = models[-1]

    # Creating a coefficient plot (optional)
    #coefficient_plot(models[1], feature_names)

    # Calibration Plot
    conversion_func, linear_model, poly = calibration_plot(model, X_train, Y_train, X_test, Y_test)

    # Getting the test dataframe for real-time predictions
    test_df = get_test_df(df)

    # Make predictions on a list of teams
    games_to_test = [
        ("IND", "PHI"),
        ("WAS", "UTA"),
        ("NYK", "DEN"),
        ("MIA", "BOS"),
        ("BKN", "MIN"),
        ("GSW", "SAC"),
        ("LAL", "CHI")    
    ]

    for (home_team, away_team) in games_to_test:
        make_prediction(home_team, away_team, test_df, scaler, model, conversion_func, linear_model, poly, ensemble = False)


    