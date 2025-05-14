import os
import pickle
import pandas as pd

# This script creates a dictionary of DataFrames from a structured dataset.
# It processes gaze feature data and groups it into a dictionary of DataFrames by 'index_row'.
# Note: This is pseudocode and should be adapted to your specific dataset and requirements.

# Step 1: Create train, test, and val CSV files, and save them in the ROOT_PATH directory.
ROOT_PATH = '...' 
READING_FEATURES_NAME = 'features.csv'


def save_dicts(my_dict, json_file_name):
    """Load a grouped Pandas DataFrame into a pickle file."""
    with open(json_file_name+'.pkl', 'wb') as json_file:
        pickle.dump(my_dict, json_file)

def create_dict_dataset(df):
    index_row = 0
    final_df = pd.DataFrame()
    for index, row in df.iterrows():
        # Step 2: Update the path to READING_FEATURES_NAME.
        # Assumes that 'gaze_path' is a column in the DataFrame.
        path = os.path.join(ROOT_PATH, row['gaze_path'])
        path_to_features = os.path.join(path, READING_FEATURES_NAME)

        if os.path.exists(path_to_features):
            current_feature_df = pd.read_csv(path_to_features)
            
            merged_df = current_feature_df.copy()
            merged_df['index_row'] = index_row
            final_df = pd.concat([final_df, merged_df], ignore_index=True)
            index_row+=1

    final_df = dict(list(final_df.groupby(['index_row'])))
    return final_df

def prepare_longitudinal_dataset():
    splitted_train_df = pd.read_csv(os.path.join(ROOT_PATH, "train.csv"))
    splitted_test_df = pd.read_csv(os.path.join(ROOT_PATH, "test.csv"))
    splitted_val_df = pd.read_csv(os.path.join(ROOT_PATH, "val.csv"))

    train = create_dict_dataset(splitted_train_df)
    test = create_dict_dataset(splitted_test_df)
    val = create_dict_dataset(splitted_val_df)
    save_dicts(train, os.path.join(ROOT_PATH, "grouped_train"))
    save_dicts(test, os.path.join(ROOT_PATH, "grouped_test"))
    save_dicts(val, os.path.join(ROOT_PATH, "grouped_val"))

if __name__ == "__main__":
    prepare_longitudinal_dataset()
