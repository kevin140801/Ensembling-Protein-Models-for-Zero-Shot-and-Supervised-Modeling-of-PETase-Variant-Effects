import os
import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import r2_score
import logging
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12576, help="Port")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for dataloader")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--hidden_layer_dim", type=int, required=True, help="Hidden layer dimension")
    parser.add_argument("--pretrained_model", type=str, default="", help="Path of pretrained model")
    parser.add_argument("--mol_dict_dir", type=str, default="", help="Path of dictionary with small molecule dictionary")
    parser.add_argument("--log_name", type=str, default="", help="Logging filename")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save the model")
    parser.add_argument("--label_dict_file", type=str, required=True, help="File with label dictionaries")
    parser.add_argument("--seq_dict_file", type=str, required=True, help="File with sequence dictionaries")
    parser.add_argument("--output_dim", type=int, required=True, help="Number of output nodes")
    parser.add_argument("--classification", type=bool, default=False, help="Classification or regression")
    parser.add_argument("--train_names", type=str, default="", help="Path to train names")
    parser.add_argument("--val_names", type=str, default="", help="Path to validation names")
    parser.add_argument("--balance_classes", type=bool, default=False, help="Balance classes")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--esm2_path", type=str, default="/gpfs/project/projects/CompCellBio/PETaseTournament/ESM2/models/esm2_t30_150M_UR50D.pt", help="Path to ESM2 model")
    parser.add_argument("--ESM2_layers", type=int, default=1, help="Number of ESM2 layers")
    parser.add_argument("--ESM2_dim", type=int, default=640, help="Dimension of ESM2")
    parser.add_argument("--pos_class_weight", type=float, default=1.0, help="Weight for positive class in classification")
    return parser.parse_args()

def get_arguments_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12576, help="Port")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for dataloader")
    parser.add_argument("--hidden_layer_dim", type=int, required=True, help="Hidden layer dimension")
    parser.add_argument("--mol_dict_dir", type=str, default="", help="Path of dictionary with small molecule dictionary")
    parser.add_argument("--log_name", type=str, default="", help="Logging filename")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save the predictions")
    parser.add_argument("--label_dict_file", type=str, required=True, help="File with label dictionaries")
    parser.add_argument("--seq_dict_file", type=str, required=True, help="File with sequence dictionaries")
    parser.add_argument("--output_dim", type=int, required=True, help="Number of output nodes")
    parser.add_argument("--classification", type=bool, default=False, help="Classification or regression")
    parser.add_argument("--test_names", type=str, required=True, help="Path to test names")
    parser.add_argument("--esm2_path", type=str, default="/gpfs/project/projects/CompCellBio/PETaseTournament/ESM2/models/esm2_t30_150M_UR50D.pt", help="Path to ESM2 model")
    parser.add_argument("--ESM2_layers", type=int, default=1, help="Number of ESM2 layers")
    parser.add_argument("--ESM2_dim", type=int, default=640, help="Dimension of ESM2")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    return parser.parse_args()

def is_cuda(device):
    return device == torch.device('cuda')

fixed_parameters = ["module.model.lm_head.layer_norm.bias",
                        "module.model.lm_head.layer_norm.weight",
                        "module.model.lm_head.dense.bias",
                        "module.model.lm_head.dense.weight",
                        "module.model.lm_head.bias",
                        "module.model.contact_head.regression.bias",
                        "module.model.contact_head.regression.weight",
                        "module.model.emb_layer_norm_after.bias",
                        "module.model.emb_layer_norm_after.weight",
                        "module.model.lm_head.weight",
                        ]

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_logging(log_name, learning_rate, num_layers, batch_size):
    setting = f"ESM2_{log_name}_lr_{learning_rate}_layers_{num_layers}_bs_{batch_size}"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filename = f"/gpfs/project/projects/CompCellBio/PETaseTournament/ESM2/logs/training/{setting}.txt"
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    fhandler = logging.FileHandler(filename=filename, mode='a')
    logger.addHandler(fhandler)
    return logger, setting


def compute_macro_f1(true_labels, logits):
    predicted_labels = np.argmax(logits, axis=1)
    return f1_score(true_labels, predicted_labels, average='macro')


def compute_fmax(y_true, y_scores, thresholds=np.linspace(0, 1, 101)):
    """
    Compute Fmax for multi-label GO term prediction.

    Parameters:
        y_true (np.ndarray): binary ground truth matrix (num_samples x num_classes)
        y_scores (np.ndarray): predicted scores matrix (num_samples x num_classes)
        thresholds (iterable): list of thresholds to evaluate

    Returns:
        float: maximum F1 score (Fmax)
    """
    fmax = 0.0
    num_samples = y_true.shape[0]

    for tau in thresholds:
        y_pred = (y_scores >= tau).astype(int)

        avg_precisions = []
        avg_recalls = []

        for i in range(num_samples):
            true_labels = y_true[i]
            pred_labels = y_pred[i]

            tp = np.sum(true_labels * pred_labels)
            fp = np.sum((1 - true_labels) * pred_labels)
            fn = np.sum(true_labels * (1 - pred_labels))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            avg_precisions.append(precision)
            avg_recalls.append(recall)

        avg_precision = np.mean(avg_precisions)
        avg_recall = np.mean(avg_recalls)

        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
            fmax = max(fmax, f1)

    return fmax
