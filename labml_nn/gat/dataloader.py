import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datacleaner import *


# custom dataset
class MIMICDataset(Dataset):
    def __init__(self, df):
        self.data = df.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = (self.data[idx][-1])
        patient = (self.data[idx][:-1])
        sample = {'patient': torch.FloatTensor(patient), 'label': torch.Tensor([label])} #torch.FloatTensor(self.data[idx])
        return sample
    

def get_cleaned_df():
    df = get_dataframe()
    
    #Count the number of diagnosis risk factors
    icd9_cols = df.filter(regex='^icd9').columns

    # Find all columns with age of diagnoses
    age_cols = df.filter(regex='admit_age').columns
    age_cols = age_cols[:-1] #don't get the HF age cause we don't want to drop that one

    # Find all columns with icu stay of diagnoses
    icu_cols = df.filter(regex='icu_stay').columns

    # Find columns associated with echo data
    echo_cols = ['height', 'weight', 'bpsys', 'bpdias', 'hr', 'EF']
    
    #Create a new dataframe that has the groupings
    grouped_df = pd.DataFrame()

    # group the diagnoses
    for i in range(len(icd9_cols)):
        # stack the three columns into a single list column
        df['new_'+icd9_cols[i]] = df.apply(lambda x: [x[icd9_cols[i]], x[age_cols[i]], x[icu_cols[i]]], axis=1)
        grouped_df['new_'+icd9_cols[i]] = (df['new_'+icd9_cols[i]])
        df.drop(('new_'+icd9_cols[i]), axis=1)
        
    #group all echo variables with age and echo_icu_stay
    for i in range(len(echo_cols)):
        # stack the three columns into a single list column
        df['new_'+echo_cols[i]] = df.apply(lambda x: [x[echo_cols[i]], x['age'], x['echo_icu_stay']], axis=1)
        grouped_df['echo_'+echo_cols[i]] = (df['new_'+echo_cols[i]])
        df.drop('new_'+echo_cols[i], axis=1)
        
    #add gender - standalone
    # replace 'M' and 'F' with 1 and 2, respectively
    df['gender'] = df['gender'].replace({'M': 1, 'F': 2})
    grouped_df['gender'] = df.apply(lambda x: [x['gender'], 0, 0], axis=1)

    #add target HF
    grouped_df['HF'] = df['target_HF']
    
    return grouped_df

def get_dataloader(batch_size, val_size, test_size):
    #Get the new data
    grouped_df = get_cleaned_df()
    
    # split data into training and test sets
    train_data, test_data = train_test_split(grouped_df, test_size=test_size)
    
    # Split the train data into train and validation sets
    train_data, val_data = train_test_split(train_data, test_size=val_size/(1-test_size))
    
    # create datasets and dataloaders for each set
    train_dataset = MIMICDataset(train_data)
    val_dataset = MIMICDataset(val_data)
    test_dataset = MIMICDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
