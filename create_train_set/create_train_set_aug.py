import glob
import random
import pandas as pd

real_perc = 100 # Number between 0% and 100%
aug_perc = 0  # Number between 0% and 100%

# Read real data
real_data = pd.read_csv('./datasets/real_training_data.csv')

# Read augmented data
negatives = glob.glob('/home/qianyu_fan/Qianyu_Code/Qianyu_Dataset/Augmentation/Negative/*')
positives = glob.glob('/home/qianyu_fan/Qianyu_Code/Qianyu_Dataset/Augmentation/Positive/*')
aug_neg = [[img_path, 0] for img_path in negatives]
aug_pos = [[img_path, 1] for img_path in positives]
augmented_data = pd.DataFrame(aug_neg+aug_pos, columns=['ID_IMG', 'BLOOD'])
augmented_data = augmented_data.sample(frac=1).reset_index(drop=True)


# Sample Real data
real_sample = real_data.sample(frac=real_perc/100).reset_index(drop=True)

# Sample Augmented data
aug_sample = augmented_data.sample(frac=aug_perc/100).reset_index(drop=True)

all_df = pd.concat([real_sample, aug_sample])[['ID_IMG','BLOOD']]
all_df = all_df.sample(frac=1).reset_index(drop=True)
all_df.to_csv(f'./datasets/train_real_{real_perc}_aug_{aug_perc}.csv')
