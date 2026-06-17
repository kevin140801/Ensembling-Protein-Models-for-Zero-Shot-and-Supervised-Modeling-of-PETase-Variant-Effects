from util_helper import *
from util_model import ESM2_FFN, load_pretrained_model
from util_data import ESM2_SM_Dataset_downstream, ESM2_Dataset_downstream
import logging
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from os.path import join
import numpy as np
import esm
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import gc

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
n_gpus = len(list(range(torch.cuda.device_count())))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = get_arguments_eval()
args_dict = vars(args)
globals().update(args_dict)

logger, setting = setup_logging(log_name, 0.00005, 0, batch_size)
labels_dict = np.load(label_dict_file, allow_pickle=True).item()
Seq_dict = np.load(seq_dict_file, allow_pickle=True).item()

if mol_dict_dir != "":
    use_sm = True
    Dataset_class = ESM2_SM_Dataset_downstream
    Mol_dict = np.load(mol_dict_dir, allow_pickle=True).item()
    small_molecule_dim = Mol_dict[list(Mol_dict.keys())[0]].shape[0]
    logging.info(f"Small molecule dimension: {small_molecule_dim}")
else:
    use_sm = False
    Dataset_class = ESM2_Dataset_downstream
    Mol_dict = None
    small_molecule_dim = 0
    logging.info("No small molecule data")


def evaluator(gpu, device, test_data):
    logging.info(f"GPU: {gpu}")
    
    if is_cuda(device):
        torch.cuda.set_device(gpu)
        setup(gpu, n_gpus, str(port))
        torch.manual_seed(0)
        
    
    # Load ESM2 model
    esm2_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_location=esm2_path)
    batch_converter = alphabet.get_batch_converter()
    
    # Create test dataset and loader
    test_dataset = Dataset_class(
        filenames=test_data,
        labels_dict=labels_dict,
        Mol_dict=Mol_dict,
        Seq_dict=Seq_dict,
        batch_converter=batch_converter,
        return_names=True
    )
    testsampler = DistributedSampler(test_dataset, shuffle=False, num_replicas=n_gpus, rank=gpu, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, sampler=testsampler, pin_memory=True)
    
    # Initialize model
    if is_cuda(device):
        model = ESM2_FFN(esm2_model, enz_dim=ESM2_dim, hidden_layer_dim=hidden_layer_dim, 
                        mol_dim=small_molecule_dim, output_dim=output_dim, ESM2_layers=ESM2_layers).to(gpu)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
    else:
        model = ESM2_FFN(esm2_model, enz_dim=ESM2_dim, hidden_layer_dim=hidden_layer_dim,
                        mol_dim=small_molecule_dim, output_dim=output_dim, ESM2_layers=ESM2_layers)
       
    
    # Load trained model
    checkpoint = torch.load(model_path)
    if "model_state_dict" in checkpoint:
        model = load_pretrained_model(model, checkpoint["model_state_dict"])
    else:
        model = load_pretrained_model(model, checkpoint)
    
    if gpu == 0:
        logging.info(f"Loading model from {model_path}")
    
    model.eval()
    predictions = []
    test_names_list = []
    
    with torch.no_grad():
        logging.info("Starting evaluation")
        for step, batch in enumerate(testloader):
            if use_sm:
                Seqs, Mols, _, names = batch
            else:
                Seqs, _, names = batch
                Mols = None
                
            labels, Sequences = Seqs[0], Seqs[1]
            Sequences = [seq[:1024] if len(seq) > 1024 else seq for seq in Sequences]
            Seqs = [(labels[i], Sequences[i]) for i in range(len(labels))]
            
            with autocast():
                batch_strs, batch_labels, X = batch_converter(Seqs)
                if is_cuda(device):
                    X = X.to(gpu, non_blocking=True)
                    if Mols is not None:
                        Mols = Mols.to(gpu, non_blocking=True)
                
                output = model(X, Mols) if use_sm else model(X)
            
            # Get predictions based on task type
            if classification:
                if output_dim == 1:
                    pred = torch.sigmoid(output).cpu().numpy()
                else:
                    pred = torch.softmax(output, dim=1).cpu().numpy()
            else:
                pred = output.cpu().numpy()
            
            predictions.extend(pred)
            test_names_list.extend(names)
            
            if gpu == 0 and (step + 1) % 10 == 0:
                logging.info(f"Processed {step + 1}/{len(testloader)} batches")
    

    if n_gpus > 1:
        world_size = torch.distributed.get_world_size()
        predictions_gathered = [None for _ in range(world_size)]
        names_gathered = [None for _ in range(world_size)]
        
        torch.distributed.all_gather_object(predictions_gathered, predictions)
        torch.distributed.all_gather_object(names_gathered, test_names_list)
        
        if gpu == 0:
            # Combine predictions and names from all GPUs
            all_predictions = []
            all_names = []
            for p, n in zip(predictions_gathered, names_gathered):
                all_predictions.extend(p)
                all_names.extend(n)
            
            # Save results
            results = {
                'predictions': np.array(all_predictions),
                'test_names': all_names
            }
            np.save(join(save_dir, log_name + '_test_predictions.npy'), results)
            logging.info(f"Results saved to {join(save_dir, log_name + '_test_predictions.npy')}")
    else:
        # Single GPU case
        results = {
            'predictions': np.array(predictions),
            'test_names': test_names_list
        }
        np.save(join(save_dir, log_name + '_test_predictions.npy'), results)
        logging.info(f"Results saved to {join(save_dir, log_name + '_test_predictions.npy')}")
    
    cleanup()

if __name__ == '__main__':
    # Load test data
    test_data = np.load(test_names, allow_pickle=True).tolist()
    #if classification and output dim > 1: check whether test names are in Seq_dict keys
    if classification and output_dim > 1:
        logging.info("Checking whether test names are in Seq_dict keys")
        logging.info("Start number: " + str(len(test_data)))
        test_names = [name for name in test_data if name in Seq_dict.keys()]
        logging.info(f"Number of test samples: {len(test_names)}")
        test_data = test_names

    logging.info(f"Number of test samples: {len(test_data)}")

    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_ids = list(range(torch.cuda.device_count()))
        gpus = len(device_ids)
        args.world_size = gpus
        
    else:
        device = torch.device('cpu')
        args.world_size = -1
    
    if torch.cuda.is_available():
            mp.spawn(evaluator, nprocs=n_gpus, args=([device, test_data]))
    else:
        evaluator(0, device, test_data)