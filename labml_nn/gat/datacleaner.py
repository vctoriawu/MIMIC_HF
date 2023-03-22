import pandas as pd
import numpy as np

def read_data():
    df = pd.read_csv('labml_nn/gat/mimic_hf.csv')
    return df

def drop_duplicates(df):
    
    # make sure we only get one of each patient
    duplicate_column = 'subject_id'
    df.drop_duplicates(subset=duplicate_column, inplace=True)
    
    # drop irrelevant columns
    columns_to_drop = ['subject_id_1', 'subject_id_2', 'SUBJECT_ID_3', 'min_row_id', 'ROW_ID', 'CHARTDATE', 'CHARTTIME', 'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID', 'ISERROR', 'subject_id_4', 'subject_id_5', 'age_1', 'TEXT', 'HADM_ID_1', 'hadm_id', ]
    df = df.drop(columns=columns_to_drop)
    
    return df

def remove_future_diagnosis(df, age_cols, echo_cols):
    #Check if age at diagnosis is less than age at HF diagnosis
    for index, row in df.iterrows():
        if pd.isna(row['HF_admit_age']):
            #check the echo age ?
            pass
        else:
            # don't use a diagnosis if it happened after the HF diagnosis
            for diagnosis_age in age_cols:
                if row[diagnosis_age] > row['HF_admit_age']:
                    column_index = df.columns.get_loc(diagnosis_age)

                    #set all values related to the diagnosis to 0 if it happened after the heart failure diagnosis
                    row.iloc[column_index - 1] = np.nan
                    row.iloc[column_index] = np.nan
                    row.iloc[column_index + 1] = np.nan

            # don't use data from the echo if it happened after HF diagnosis
            if row['age'] > row['HF_admit_age']:
                for col in echo_cols:
                    row[col] = np.nan
                    
    return df

def one_hot_data(df, icd9_cols, age_cols, icu_cols, echo_cols):
    for col in icd9_cols:
        df[col] = df[col].mask(df[col].notnull(), 1)
        df[col].fillna(0, inplace=True)

    for col in age_cols:
        df[col].fillna(-1, inplace=True)

    for col in icu_cols:
        df[col].fillna(-1, inplace=True)

    for col in echo_cols:
        df[col].fillna(-1, inplace=True)
        
    return df

def cleanup_EF_HF(df):
    #Rename EF and clean that column
    df.rename(columns={'V1': 'EF'}, inplace=True)
    df['EF'].fillna(-1, inplace=True)

    # Clean HF data columns
    df['target_HF'] = df['target_HF'].mask(df['target_HF'].notnull(), 1)
    df['target_HF'].fillna(0, inplace=True)
    df['HF_admit_age'].fillna(-1, inplace=True)
    
    return df
    
                
def get_dataframe():
    df = read_data()
    df = drop_duplicates(df)
    
    column_name = 'target_HF'
    nan_mask = df[column_name].isna()
    num_nan = nan_mask.sum()
    print(f'The number of negative values in {column_name} column is: {num_nan}')
    print(f'The number of positive values in {column_name} column is: {len(df) - num_nan}')
    
    #Count the number of diagnosis risk factors
    icd9_cols = df.filter(regex='^icd9').columns
    
    hf_age = df['HF_admit_age'] 

    # Find all columns with age of diagnoses
    age_cols = df.filter(regex='admit_age').columns
    age_cols = age_cols[:-1] #don't get the HF age cause we don't want to drop that one

    # Find all columns with icu stay of diagnoses
    icu_cols = df.filter(regex='icu_stay').columns

    # Find columns associated with echo data
    echo_cols = ['height', 'weight', 'bpsys', 'bpdias', 'hr']
    
    df = remove_future_diagnosis(df, age_cols, echo_cols)
    
    df = one_hot_data(df, icd9_cols, age_cols, icu_cols, echo_cols)
    
    df = cleanup_EF_HF(df)
    
    return df
    
    