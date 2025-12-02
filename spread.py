from util import *

# FOR A DRY RUN FOR WAGERING: 
# Load new games -> True 
# Set the test_size to smallest possible setting 

if __name__ == "__main__":
    # Loading Games
    df = load_data(load_new_games = False, start_date='10/01/1994')

    print("Processing Dataset...")
    # Preprocessing dataset
    df = filter_dataset(df)

    # Grouping the data by team & season
    running_totals = groupby_team_season_spread(df)

    # Matching teams to their opponents
    match_df = match_opponents_optimized(running_totals)

    print("Preprocessing Finished!\n")

    # Split into training and validation sets
    #X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training(match_df, test_size=2/len(match_df), random_state=420)
    X_train, X_test, Y_train, Y_test, feature_names, scaler = preprocess_training_spread(match_df, test_size=0.20, random_state=420)

    # Train models
    print("Training Models...")
    model = MLPRegressor(
        (25), activation='relu', alpha=1e-4, verbose=True, 
        max_iter=1000, tol=1e-5, n_iter_no_change=50
    )
    model = RidgeCV(alphas=np.logspace(-6,3,100))
    model.fit(X_train, Y_train)
    Yhat_test = np.round(model.predict(X_test))
    print("Test MAE: %.4f" % (mean_absolute_error(Y_test, Yhat_test)))
    

    plt.figure(figsize=(8,8))
    plt.scatter(Y_test, Yhat_test, s = 1.0)
    sns.kdeplot(x=Y_test, y=Yhat_test, cmap="Blues", fill=True, bw_adjust = 0.75)
    plt.grid()
    plt.show()

    print("Models Trained!\n")

    # Getting the test dataframe for real-time predictions
    test_df = get_test_df(df, 2024)

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
    results_df.to_excel("data/Results.xlsx", index=False)