from torch.utils.data import Dataset
import numpy as np
import torch

class ESM2_SM_Dataset_downstream(Dataset):
    def __init__(self,  
                 filenames,
                 labels_dict,
                 Mol_dict,
                 Seq_dict,
                 batch_converter,
                 return_names = False):
        
        self.total_datacount = len(filenames)
        self.filenames = filenames
        self.labels_dict = labels_dict
        self.Mol_dict = Mol_dict
        self.Seq_dict = Seq_dict
        self.batch_converter = batch_converter
        self.return_names = return_names

    def __len__(self):
        return self.total_datacount

    def __getitem__(self, idx):
        name = self.filenames[idx]
        UID = name.split("_")[0]
        MID = name.split("_")[1]
        y = self.labels_dict[name]
        y = torch.tensor(y)
        mol = np.array(self.Mol_dict[MID])
        mol = torch.tensor(mol, dtype=torch.float32)  
        seq = self.Seq_dict[UID]
        # Truncate sequence if longer than 1024
        if len(seq) > 1024:
            seq = seq[:1024]
        if self.return_names:
            return (UID, seq), mol, y, name
        return (UID, seq), mol, y


class ESM2_Dataset_downstream(Dataset):
    def __init__(self,  
                 filenames,
                 labels_dict,
                 Seq_dict,
                 batch_converter,
                 Mol_dict = None,
                 return_names = False):
        
        self.total_datacount = len(filenames)
        self.filenames = filenames
        self.labels_dict = labels_dict
        self.Seq_dict = Seq_dict
        self.batch_converter = batch_converter
        self.return_names = return_names

    def __len__(self):
        return self.total_datacount

    def __getitem__(self, idx):
        name = self.filenames[idx]
        UID = name if name in self.Seq_dict else name.split("_")[0]
        y = self.labels_dict[name]
        y = torch.tensor(y)
        seq = self.Seq_dict[UID]
        # Truncate sequence if longer than 1024
        if len(seq) > 1024:
            seq = seq[:1024]
        if self.return_names:
            return (UID, seq), y, name
        return (UID, seq), y