import torch
import torch.nn as nn
import logging

class ESM2_FFN(torch.nn.Module):
    def __init__(
            self, 
            model,
            enz_dim,
            hidden_layer_dim,
            mol_dim,
            output_dim, 
            ESM2_layers
            ):
        super(ESM2_FFN, self).__init__()

        self.model = model
        self.ESM2_layers = ESM2_layers
                    
        self.small_molecule_dim = mol_dim
        self.enz_dim = enz_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.enz_dim + self.small_molecule_dim, self.hidden_layer_dim, bias=True),
            nn.BatchNorm1d(self.hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_layer_dim, self.output_dim),
        )
        
    @torch.cuda.amp.autocast()
    def forward(self, batch_tokens, mol = None):
        representations = self.model(
            batch_tokens, 
            repr_layers=[self.ESM2_layers],  
            return_contacts=False
        )
        x = representations["representations"][self.ESM2_layers][:, 0, :]
        x = torch.flatten(x, 1)
        if mol is not None:
            x = torch.cat((x, mol), 1)
        results = self.fc(x)
        return results


def load_pretrained_model(current_model, pretrained_state_dict):
    current_state_dict = current_model.state_dict()
    new_state_dict = {}
    
    for name, param in pretrained_state_dict.items():
        #name = name.replace("module.", "")
        if not name.startswith("module."):
            name = "module." + name
        if name in current_state_dict:
            if current_state_dict[name].shape == param.shape:
                new_state_dict[name] = param
            else:
                logging.info(f"Skipping parameter {name} due to shape mismatch.")
        else:
            logging.info(f"Parameter {name} not found in current model.")

    current_model.load_state_dict(new_state_dict, strict=False)
    return current_model