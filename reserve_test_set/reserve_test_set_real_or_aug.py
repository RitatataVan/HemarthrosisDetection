import glob
import random
import pandas as pd

negatives = glob.glob('/home/qianyu_fan/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Negative/*')
positives = glob.glob('/home/qianyu_fan/Qianyu_Code/Qianyu_Dataset/Real_Hemophilia/Positive/*')

random.shuffle(negatives)
random.shuffle(positives)

# Splitting Images
test_negatives = negatives[:100]
train_negatives = negatives[100:]
test_positives = positives[:100]
train_positives = positives[100:]

# Saving the testing set
test_neg = [[img_path, 0] for img_path in test_negatives]
test_pos = [[img_path, 1] for img_path in test_positives]

test_df = pd.DataFrame(test_neg+test_pos, columns=['ID_IMG', 'BLOOD'])
test_df = test_df.sample(frac=1).reset_index(drop=True)
test_df.to_csv('./datasets/real_testing_data.csv')
print('\nTesting Sample:\n', test_df)

# Saving the training set
train_neg = [[img_path, 0] for img_path in train_negatives]
train_pos = [[img_path, 1] for img_path in train_positives]

train_df = pd.DataFrame(train_neg+train_pos, columns=['ID_IMG', 'BLOOD'])
train_df = train_df.sample(frac=1).reset_index(drop=True)
print('Training Sample:\n', train_df)
train_df.to_csv('./datasets/real_training_data.csv')
