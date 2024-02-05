from functions import *

load_new_games = False

if __name__ == "__main__":
    # Loading Games
    print("Loading Games...")
    if load_new_games:
        df = get_games()        
        df.to_csv("RawDF.csv", index=False)
        df = pd.read_csv("RawDF.csv")
    else:
        df = pd.read_csv("RawDF.csv")
    
    print("Loaded Games!\n")

    print("Processing Dataset...")
    # Preprocessing dataset
    df = filter_dataset(df)

    # Grouping the data by team & season
    running_totals = groupby_team_season(df)

    # Matching teams to their opponents
    match_df = match_opponents_optimized(running_totals)

    print("Preprocessing Finished!\n")

    # Split into training and validation sets
    #X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=2/len(match_df), random_state=420)
    X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=0.20, random_state=70)

    # Train models
    print("Training Models...")
    models, weights = train_models(X_train, Y_train, mlp=True, logit=False, knn=False, rf=False, gb = False)
    print("Models Trained!\n")

    # Creating an ensemble model
    #model = EnsembleMax(models, weights, 0.501, 0.499)
    model = models[0]

    evaluate_model(model, X_train, X_test, Y_train, Y_test)

    # Creating a coefficient plot (optional)
    #coefficient_plot(models[0], feature_names)

    # Calibration Plot
    conversion_func, linear_model, poly = calibration_plot(model, X_train, Y_train, X_test, Y_test)

    # Getting the test dataframe for real-time predictions
    test_df = get_test_df(df)

    # Make predictions on a list of teams
    games_to_test = [
        ("SAC", "CLE"),
        ("CHA", "LAL"),
        ("PHI", "DAL"),
        ("GSW", "BKN"),
        ("ATL", "LAL"),
        ("TOR", "NOP")
    ]

    for (home_team, away_team) in games_to_test:
        make_prediction(home_team, away_team, test_df, scaler, model, conversion_func, linear_model, poly, ensemble = False)


    