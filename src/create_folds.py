import os
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from config_file import config

def main():
    df = pd.read_csv(os.path.join(config['data_dir'], 'train.csv'))
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=config['n_splits'])
    for fold_, (train_idx, valid_idx) in tqdm(enumerate(kf.split(X=df, y=y)), total=config['n_splits']):
        df.loc[valid_idx, 'kfold'] = fold_
        print ('fold %d'%fold_)
    df.to_csv(os.path.join(config['data_dir'], 'train_folds.csv'))

if __name__ == "__main__":
    main()