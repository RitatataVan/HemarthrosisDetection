import glob
import random
import pandas as pd

real_perc = 100 # Number between 0% and 100%
syn_perc = 0  # Number between 0% and 100%

# Read real data
real_data = pd.read_csv('./datasets/real_training_data.csv')

# Read synthetic data
negatives_syn = glob.glob('/home/qianyu_fan/Qianyu_Code/Qianyu_Dataset/Synthetic_Hemophilia/Negative/*')
positives_syn = glob.glob('/home/qianyu_fan/Qianyu_Code/Qianyu_Dataset/Synthetic_Hemophilia/Positive/*')

syn_train_negatives = negatives_syn[50:]
syn_train_positives = positives_syn[50:]
syn_neg = [[img_path, 0] for img_path in syn_train_negatives]
syn_pos = [[img_path, 1] for img_path in syn_train_positives]
synthetic_data = pd.DataFrame(syn_neg+syn_pos, columns=['ID_IMG', 'BLOOD'])
synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)

# Sample Real data
real_sample = real_data.sample(frac=real_perc/100).reset_index(drop=True)

# Sample Synthetic data
syn_sample = synthetic_data.sample(frac=syn_perc/100).reset_index(drop=True)

all_df = pd.concat([real_sample, syn_sample])[['ID_IMG','BLOOD']]
all_df = all_df.sample(frac=1).reset_index(drop=True)
all_df.to_csv(f'./datasets/train_real_{real_perc}_syn_{syn_perc}.csv')
