from util_helper import *
from util_model import ESM2_FFN, load_pretrained_model
from util_data import ESM2_SM_Dataset_downstream, ESM2_Dataset_downstream
import argparse
import logging
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from os.path import join
import numpy as np
import esm
import time
from sklearn.metrics import accuracy_score, matthews_corrcoef, r2_score
import torch
import torch.nn as nn
import torch.multiprocessing as mp  
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc
import random

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  
torch.backends.cudnn.allow_tf32 = True 
n_gpus = len(list(range(torch.cuda.device_count())))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_arguments()
args_dict = vars(args)
globals().update(args_dict)

logger, setting = setup_logging(log_name, learning_rate, 0, batch_size)
labels_dict = np.load(label_dict_file, allow_pickle=True).item()
Seq_dict = np.load(seq_dict_file, allow_pickle=True).item()

if mol_dict_dir != "":
    use_sm = True
    Dataset_class = ESM2_SM_Dataset_downstream
    Mol_dict = np.load(mol_dict_dir, allow_pickle=True).item()
    small_molecule_dim = Mol_dict[list(Mol_dict.keys())[0]].shape[0]
    logging.info("Small molecule dimension: " + str(small_molecule_dim))
else:
    use_sm = False
    Mol_dict = None
    Dataset_class = ESM2_Dataset_downstream
    small_molecule_dim = 0 
    logging.info("No small molecule data")


def trainer(gpu, device, train_data, val_data, all_UIDs, all_MIDs):
    
    logging.info("GPU: " + str(gpu) + "Port: " + str(port))

    if is_cuda(device):
        torch.cuda.set_device(gpu) 
        setup(gpu, n_gpus, str(port))
        torch.manual_seed(0)
        
    torch.cuda.empty_cache()
    gc.collect()
    
    esm2_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location = esm2_path)
    batch_converter = alphabet.get_batch_converter()
    
    if is_cuda(device):
        model = ESM2_FFN(esm2_model, enz_dim=ESM2_dim, hidden_layer_dim=hidden_layer_dim, mol_dim=small_molecule_dim,
                    output_dim=output_dim, ESM2_layers=ESM2_layers).to(gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        logging.info("Model moved to GPU")
    else:
        model = ESM2_FFN(esm2_model, enz_dim=ESM2_dim, hidden_layer_dim=hidden_layer_dim, mol_dim=small_molecule_dim,
                    output_dim=output_dim, ESM2_layers=ESM2_layers)


    val_dataset = Dataset_class(
                    filenames = val_data,
                    labels_dict = labels_dict,
                    Mol_dict = Mol_dict,
                    Seq_dict = Seq_dict,
                    batch_converter = batch_converter
                    )
    valsampler = DistributedSampler(val_dataset, shuffle = False, num_replicas = n_gpus, rank = gpu, drop_last = True)
    valloader = DataLoader(val_dataset, batch_size=batch_size,  num_workers=num_workers, sampler=valsampler, pin_memory=True)


    for name, param in model.named_parameters():
        if name in fixed_parameters:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if not param.requires_grad:
            logging.info(f"Parameter {name} is frozen.")


    if pretrained_model != "":
        checkpoint = torch.load(pretrained_model)
        if "model_state_dict" in checkpoint:
            model = load_pretrained_model(model, checkpoint["model_state_dict"])
        else:
            model = load_pretrained_model(model, checkpoint)

    if gpu == 0:
        logging.info("Number of trainable parameters: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if classification:
        if output_dim == 1:
            pos_weight = torch.tensor([pos_class_weight]).to(gpu) if is_cuda(device) else torch.tensor([pos_class_weight])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            if pos_class_weight != 1:
                train_data = np.load(train_names, allow_pickle=True).tolist()
                keys = list(combined_bounding_boxes.keys())
                train_data = list(set(train_data).intersection(keys))
                count_class = [0] * output_dim
                for x in train_data:
                    count_class[labels_dict[x]] += 1
                weights = 1 / torch.tensor(count_class)
                weights = weights / weights.max()
                if is_cuda(device):
                    weights = weights.to(gpu)
                logging.info("Class weights: " + str(weights))
                criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    if pretrained_model != "":
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    val_loss_old = -float('inf')

    scaler = GradScaler()
    logging.info("Starting training:")
    for epochs in range(num_epochs):

        if classification and balance_classes and output_dim == 1:
            train_data = np.load(train_name, allow_pickle=True).tolist()
            train_data = [x for x in train_data if x.split("_")[0] in all_UIDs and x.split("_")[1] in all_MIDs]
            train_data_pos = [x for x in train_data if labels_dict[x] == 1]
            train_data_neg = [x for x in train_data if labels_dict[x] == 0]
            train_data_neg = list(np.random.choice(train_data_neg, len(train_data_pos), replace=False))
            train_data = train_data_pos + train_data_neg
        elif classification and balance_classes and output_dim > 1:
            train_data = np.load(train_names, allow_pickle=True).tolist()
            keys = list(combined_bounding_boxes.keys())
            train_data = list(set(train_data).intersection(keys))
            count_class = [0] * output_dim
            for x in train_data:
                count_class[labels_dict[x]] += 1
                #make sure the smallest classest are in the worst case 10 times underrepresented
            min_count = min(count_class) * 10
            new_train_data = []
            for i in range(output_dim):
                class_indices = [j for j, label in enumerate(train_data) if labels_dict[label] == i]
                new_train_data += list(np.random.choice(class_indices, min(min_count, len(class_indices)), replace=False))
            
            train_data = [train_data[i] for i in new_train_data]
            random.shuffle(train_data)

        #if "IC50" in train_names:
        #    train_data = train_data[:50000]

        train_dataset = Dataset_class(
                    filenames = train_data,
                    labels_dict = labels_dict,
                    Mol_dict = Mol_dict,
                    Seq_dict = Seq_dict,
                    batch_converter = batch_converter
                    )
        trainsampler = DistributedSampler(train_dataset, shuffle = True, num_replicas = n_gpus, rank = gpu, drop_last = True)
        trainloader = DataLoader(train_dataset, batch_size=batch_size,  num_workers=num_workers, sampler=trainsampler, pin_memory=True)

        logging.info("Train data loaded")
        logging.info("Length of trainloader: " + str(len(trainloader)))

        epoch_time = time.perf_counter()
        model.train()

        train_loss = 0
        y_true, y_pred = [], []

        for step, batch in enumerate(trainloader):
            if use_sm:
                Seqs, Mols, y = batch
            else:
                Seqs, y = batch
                Mols = None
            labels, Sequences = Seqs[0], Seqs[1]
            Sequences = [seq[:1024] if len(seq) > 1024 else seq for seq in Sequences]
            Seqs = [(labels[i], Sequences[i]) for i in range(len(labels))]
            

            with torch.cuda.amp.autocast():
                batch_strs, batch_labels, X = batch_converter(Seqs)
                if is_cuda(device):
                    X = X.to(gpu, non_blocking=True)
                    if use_sm:
                        Mols = Mols.to(gpu, non_blocking=True)
                    y = y.to(gpu, non_blocking=True)
                    
                del Seqs, batch_strs, batch_labels
                
                output = model(X, Mols)
                y, output = y.float(), output.float()
                if classification and output_dim > 1:
                    loss = criterion(output.view(-1, output_dim).float(), y.view(-1, output_dim).float())
                else:
                    loss = criterion(output.view(-1), y.view(-1))

                scaled_loss = loss / gradient_accumulation_steps 
                scaler.scale(scaled_loss).backward()
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                del X, Mols
                torch.cuda.empty_cache()
            
            if step % 100 == 0:
                torch.cuda.synchronize()
                
            train_loss += loss.item()  # Use original loss for accumulation

            if classification and output_dim == 1:
                output = torch.sigmoid(output)

            if classification and output_dim > 1:
                y_true.extend(y.detach().cpu().numpy().reshape(-1, output_dim)), y_pred.extend(output.detach().cpu().numpy().reshape(-1, output_dim))
            else:
                y_true.extend(y.detach().cpu().numpy().reshape(-1)), y_pred.extend(output.detach().cpu().numpy().reshape(-1))
            
            if step % 20 == 0:
                y_true, y_pred = y_true[-500:], y_pred[-500:]
                if classification and output_dim == 1:
                    train_mcc = matthews_corrcoef(y_true, np.round(y_pred))
                    train_acc = accuracy_score(y_true, np.round(y_pred))
                    logging.info("Epoch: " + str(epochs) + " Step: " + str(step) + " Loss: " + str(train_loss/(step+1)) + \
                                    " MCC: " + str(train_mcc) + "ACC: " + str(train_acc))
                elif classification and output_dim > 1:
                    val_fmax  = compute_fmax(np.array(y_true), np.array(y_pred))
                    logging.info("Epoch: " + str(epochs) + " Step: " + str(step) + " Loss: " + str(train_loss/(step+1)) + \
                                    " Fmax: " + str(val_fmax))
                else:
                    train_mse = np.mean((np.array(y_true) - np.array(y_pred))**2)
                    if output_dim ==1:
                        train_r2 = r2_score(y_true, y_pred)
                    else:
                        train_r2 = compute_average_r2(y_true, y_pred, output_dim)
                    logging.info("Epoch: " + str(epochs) + " Step: " + str(step) + " Loss: " + str(train_loss/(step+1)) + \
                                    " MSE: " + str(train_mse) + " R2: " + str(train_r2))
                
        torch.cuda.empty_cache()
        gc.collect()

                
        logging.info("Epoch: " + str(epochs) + " Loss: " + str(train_loss/(step+1)))
        if gpu == 0:
            epoch_time = time.perf_counter() - epoch_time
            logging.info("Epoch time: " + str(epoch_time))


        model.eval()
        with torch.no_grad(), autocast():
            val_loss = 0
            y_true, y_pred = [], []
            for step, batch in enumerate(valloader):
                if use_sm:
                    Seqs, Mols, y = batch
                else:
                    Seqs, y = batch
                    Mols = None
                labels, Sequences = Seqs[0], Seqs[1]
                Sequences = [seq[:1024] if len(seq) > 1024 else seq for seq in Sequences]
                Seqs = [(labels[i], Sequences[i]) for i in range(len(labels))]
                
                batch_strs, batch_labels, X = batch_converter(Seqs)

                if is_cuda(device):
                    X = X.to(gpu, non_blocking=True)
                    if use_sm:
                        Mols = Mols.to(gpu, non_blocking=True)
                    y = y.to(gpu, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    output = model(X, Mols)  
                    y, output = y.float(), output.float()
                    if classification and output_dim > 1:
                        loss = criterion(output.view(-1, output_dim).float(), y.view(-1, output_dim).float())
                    else:
                        loss = criterion(output.view(-1), y.view(-1))
                    val_loss += loss.item()

                if classification and output_dim == 1:
                    output = torch.sigmoid(output)
                
                if classification and output_dim > 1:
                    y_true.extend(y.detach().cpu().numpy().reshape(-1, output_dim)), y_pred.extend(output.detach().cpu().numpy().reshape(-1, output_dim))
                else:
                    y_true.extend(y.detach().cpu().numpy().reshape(-1)), y_pred.extend(output.detach().cpu().numpy().reshape(-1))

            if classification and output_dim == 1:
                val_mcc = matthews_corrcoef(y_true, np.round(y_pred))
                val_acc = accuracy_score(y_true, np.round(y_pred))
                logging.info("Validation Loss: " + str(val_loss/(step+1)) + " Validation MCC: " + str(val_mcc) + " Validation ACC: " + str(val_acc) + " GPU: " + str(gpu))
            elif classification and output_dim > 1:
                val_fmax  = compute_fmax(np.array(y_true), np.array(y_pred))
                logging.info("Validation Loss: " + str(val_loss/(step+1)) + " Validation Fmax: " + str(val_fmax) + " GPU: " + str(gpu))
            else:
                val_mse = np.mean((np.array(y_true) - np.array(y_pred))**2)
                if output_dim == 1:
                    val_r2 = r2_score(y_true, y_pred)
                else:
                    val_r2 = compute_average_r2(y_true, y_pred, output_dim)
                logging.info("Validation Loss: " + str(val_loss/(step+1)) + " Validation MSE: " + str(val_mse) + " Validation R2: " + str(val_r2) + " GPU: " + str(gpu))

            torch.cuda.empty_cache()
            gc.collect()

        if classification:
            if output_dim == 1:
                if val_mcc > val_loss_old:
                    logging.info("New best model found")
                    val_loss_old = val_mcc
                    if gpu == 0:
                        torch.save(model.module.state_dict(), join(save_dir, setting + "_best_model.pth"))
            else:
                if val_fmax > val_loss_old:
                    logging.info("New best model found")
                    val_loss_old = val_fmax
                    if gpu == 0:
                        torch.save(model.module.state_dict(), join(save_dir, setting + "_best_model.pth"))
        else:
            if val_r2 > val_loss_old:
                logging.info("New best model found")
                val_loss_old = val_r2
                if gpu == 0:
                    torch.save(model.module.state_dict(), join(save_dir, setting + "_best_model.pth"))


            
if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    logging.info("Output dim: " + str(output_dim))
    logging.info("Classification: " + str(classification))
    logging.info("Balance classes: " + str(balance_classes))
    logging.info("Learning rate: " + str(learning_rate))
    logging.info("Batch size: " + str(batch_size))
    logging.info("Number of workers: " + str(num_workers))
    logging.info("Number of epochs: " + str(num_epochs))
    logging.info("Hidden layer dimension: " + str(hidden_layer_dim))
    logging.info("Pretrained model: " + str(pretrained_model))

    input_data = list(labels_dict.keys())
    logging.info("Number of input files: " + str(len(input_data)))

    train_data = np.load(train_names, allow_pickle=True).tolist()
    val_data = np.load(val_names, allow_pickle=True).tolist()

    all_UIDs = set(Seq_dict.keys())
    if use_sm:
        all_MIDs = set(Mol_dict.keys())
        train_data = [x for x in train_data if x.split("_")[0] in all_UIDs and x.split("_")[1] in all_MIDs]
        val_data = [x for x in val_data if x.split("_")[0] in all_UIDs and x.split("_")[1] in all_MIDs]
    else:
        all_MIDs = None
        train_data = [x for x in train_data if x in all_UIDs]
        val_data = [x for x in val_data if x in all_UIDs]
    
    logging.info("Train data size: " + str(len(train_data)), "Validation data size: " + str(len(val_data)))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = len(device_ids)
        args.world_size = gpus
        
    else:
        device = torch.device('cpu')
        args.world_size = -1
  
    try:
        if torch.cuda.is_available():
            mp.spawn(trainer, nprocs=n_gpus, args=([device, train_data, val_data, all_UIDs, all_MIDs]))
        else:
            trainer(0, [device, train_data, val_data, all_UIDs, all_MIDs])
    except Exception as e:
        print(e)