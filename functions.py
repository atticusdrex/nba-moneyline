from sportsreference.nba.teams import Teams
import pandas as pd
from datetime import datetime
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def get_games(start_date = '10/01/1994'):
    games_df = pd.DataFrame()

    nba_teams = teams.get_teams()

    for team in tqdm(nba_teams):
        team_id = team['id']
        # Query for games where the Celtics were playing
        gamefinder = leaguegamefinder.LeagueGameFinder(date_from_nullable=start_date,team_id_nullable=team_id)
        # The first DataFrame of those returned is what we want.
        games = gamefinder.get_data_frames()[0]
        
        games_df = pd.concat((games_df, games))

    return games_df

def prob_plot(Y_true, Y_pred, bins=10):
    successes = np.zeros((bins))
    totals = np.zeros_like(successes)
    bounds = zip(np.linspace(0, 1, bins + 1)[0:-1], np.linspace(0,1,bins+1)[1:])
    print(bounds)

    for (lower, upper) in bounds:
        print(lower)
        print(upper)
        break
        
import matplotlib.pyplot as plt
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
            probs[i] = 0
            samples[i] = total / len(Y_pred)
            errors[i] = 1.0
    
    return (bound_vec, probs, samples, errors)

def conversion_func(x):
    return 0.33 - 3.24044818*x+19.26369605*x**2-41.33864052*x**3+42.70963672*x**4-17.06383276*x**5