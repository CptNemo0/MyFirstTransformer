from classes.dataset import Dataset
from classes.loader import Loader
from classes.model import Transformer
from classes.trainer import Trainer

import torch
from torch.cuda import is_available
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

#Shared consts
CONTEXT_LEN     = 50
BATCH_SIZE      = 4
NUM_EMBEDDINGS  = 100288
DEVICE          = 'cuda' if is_available() else 'cpu'

#Dataset settings
DATASET_PATH    = 'datasets\\mickiewicz.txt'

#Transformer settings
BIAS            = True
D_FF            = 768 * 4 
EMBEDDING_DIM   = 768
N_HEADS         = 12 
N_LAYERS        = 12
PARAMS_PATH     = None
P_DORP          = 0.1

#Optimizer settings
LEARNING_RATE   = 0.001

#Scheaduler settings
STEP_SIZE       = 750
GAMMA           = 0.5

#   Trainer settings
EPOCHS          = 21300 * 4 * 2
LOG_INTERVAL    = 50
SAVE_INTERVAL   = 100

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():

    #Data objects
    dataset = Dataset(DATASET_PATH, CONTEXT_LEN)
    loader = Loader(dataset.get_dataset('train'), BATCH_SIZE, CONTEXT_LEN, NUM_EMBEDDINGS)

    #Model
    Leia = Transformer(NUM_EMBEDDINGS, CONTEXT_LEN, N_LAYERS, EMBEDDING_DIM, D_FF, N_HEADS, BIAS, P_DORP, DEVICE)
    #Loading params
    if PARAMS_PATH is not None:
        Leia.load_state_dict(torch.load(PARAMS_PATH))

    print(count_parameters(Leia))

    #Training objects
    loss_function = nn.CrossEntropyLoss()
    optimizer = AdamW(Leia.parameters(), LEARNING_RATE)
    scheaduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    trainer = Trainer(loader, Leia, loss_function, optimizer, DEVICE)
    trainer.train(EPOCHS, LOG_INTERVAL, SAVE_INTERVAL)

if __name__ == "__main__":
    main()