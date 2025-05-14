import utils
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def prepare_features(df):
    """Prepares and cleans the features from a pandas DataFrame for model input."""
    columns_to_drop = [
        'win_w', 'win_h', 'rad_name',#'win_width', 'win_height', 'radiologist',
        'case', 'fname',
        'label'
    ]
    df = df.drop(columns_to_drop, axis=1)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'folder_num' in df.columns:
        df = df.drop(['folder_num'], axis=1)
    if {'elapsed_time', 'start_time'}.issubset(df.columns):
        df = df.drop(['elapsed_time', 'start_time'], axis=1)
    if 'timestamp' in df.columns:
        df = df.drop(['timestamp'], axis=1)
    if {'speed_mean', 'speed_std'}.issubset(df.columns):
        df = df.drop(['speed_mean', 'speed_std'], axis=1)
    if {'acceleration_mean', 'acceleration_std'}.issubset(df.columns):
        df = df.drop(['acceleration_mean', 'acceleration_std'], axis=1)
    if {'x', 'y'}.issubset(df.columns):
        df = df.drop(['x', 'y'], axis=1)
    if {'x0', 'y0'}.issubset(df.columns):
        df = df.drop(['x0', 'y0'], axis=1)
    if 'rad_name' in  df.columns:
        df = df.drop(['rad_name'], axis=1)
    if 'doctor_confidence' in  df.columns:
        df = df.drop(['doctor_confidence'], axis=1)
    if 'index_row' in  df.columns:
        df = df.drop(['index_row'], axis=1)
    if 'gaze_path' in  df.columns:
        df = df.drop(['gaze_path'], axis=1)
    if 'folder_num' in  df.columns:
        df = df.drop(['folder_num'], axis=1)
    return torch.tensor(df.values, dtype=torch.float32)

def return_reading_seq(seq, sequence_length):
    if len(seq) < sequence_length:
        pad_length = sequence_length - len(seq)
        seq = torch.cat((seq, torch.full((pad_length, seq.shape[1]), -1, dtype=torch.float32)))
    elif len(seq) > sequence_length:
        seq = seq[-sequence_length:]
    return seq

class ViTabularDataset(Dataset):
    def __init__(self, grouped_data, 
                    sequence_length, 
                    normalization, 
                    data_folder,
                    path_to_patches):
        self.sequences = []
        self.masks = []
        self.patches_array = []
        self.labels = []
        self.emb = []

        for i, (prefix, data) in enumerate(grouped_data.items()):
            data = data.reset_index(drop=True)
            y_main = torch.tensor(data['label'].values, dtype=torch.float32)
            x_main = prepare_features(data)
            seq = return_reading_seq(x_main, sequence_length)
            self.sequences.extend(seq.unsqueeze(0))
            self.labels.extend(y_main[0].unsqueeze(0))

            patches, patch_index = self.get_anatomical_patches(data, normalization)

            self.masks.extend(patch_index)
            self.patches_array.extend(patches)

    def get_anatomical_patches(self, data, normalization):
        data = data.reset_index(drop=True)
        data.fillna(-1, inplace=True)
        if data['case'][0] == 400:
            PATH = data_folder + '400/' + data['gaze_path'][0][:-8]
        else:
            PATH = data_folder + data['gaze_path'][0][:-13]
        
        patches = torch.from_numpy(np.load(path_to_patches + data['fname'][0] + '.npy')).float()
        if len(patches) > 97:
            print(data['fname'])
            raise ValueError("This is an error to stop the code.")
        
        patch_index =  torch.from_numpy(data['patch_number'].values).int()
        if len(patch_index) < len(patches):
            padding_samples = len(patches) - len(patch_index)
            patch_index = F.pad(patch_index, (0,padding_samples), 'constant', -1)
        else:
            patch_index = patch_index[-len(patches):]
        
        patch_index = patch_index.int()

        if normalization:
            patches = torch.stack([patch / 255 for patch in patches])
        
        return patches.unsqueeze(0), patch_index.unsqueeze(0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.patches_array[idx], self.masks[idx], self.labels[idx]

def load_data(**kwargs):
    data_folder = kwargs.get("data_folder", " ")
    path_to_patches = kwargs.get("path_to_patches", ".../__allData/miccai2025/anatomical_patches_16_img224")
    shuffle = kwargs.get("shuffle", False)
    sequence_length = kwargs.get("seq_length", 97)
    batch_size = kwargs.get("batch_size", 64)
    normalization = kwargs.get("normalization", False)

    grouped_test = utils.load_pickle_file('/home/csn801/JBHI/lib/grouped_test.pkl')
    grouped_val = utils.load_pickle_file('/home/csn801/JBHI/lib/grouped_val.pkl')
    grouped_train = utils.load_pickle_file('/home/csn801/JBHI/lib/grouped_train.pkl')

    train_dataset = ViTabularDataset(grouped_train, 
                                     sequence_length, 
                                     normalization, 
                                     data_folder,
                                     path_to_patches)
    val_dataset = ViTabularDataset(grouped_val, 
                                   sequence_length, 
                                   normalization, 
                                   data_folder,
                                   path_to_patches)
    test_dataset = ViTabularDataset(grouped_test, 
                                    sequence_length, 
                                    normalization, 
                                    data_folder,
                                    path_to_patches)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    return train_loader, val_loader, test_loader 