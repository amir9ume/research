import optuna
from sklearn.externals import joblib
study = joblib.load('./tuned_models/optuna_1')
df = study.trials_dataframe()
print(df.head(3))



print("Best trial:")
print(study.best_trial)
