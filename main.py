from classes.dataset import Dataset
from classes.loader import Loader
from classes.model import Transformer
from classes.trainer import Trainer

import torch
from torch.cuda import is_available
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import yaml

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    path = 'settings_files\\leia-s.yaml'

    with open(path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)

    #Shared consts
    CONTEXT_LEN     = data['CONTEXT_LEN']
    BATCH_SIZE      = data['BATCH_SIZE']
    NUM_EMBEDDINGS  = data['NUM_EMBEDDINGS']
    DEVICE          = 'cuda' if is_available() else 'cpu'

    #Dataset settings
    DATASET_PATH    = data['DATASET_PATH']

    #Transformer settings
    BIAS            = data['BIAS']
    D_FF            = data['D_FF']
    EMBEDDING_DIM   = data['EMBEDDING_DIM']
    N_HEADS         = data['N_HEADS']
    N_LAYERS        = data['N_LAYERS']
    PARAMS_PATH     = data['PARAMS_PATH']
    P_DROP          = data['P_DROP']

    #Optimizer settings
    LEARNING_RATE   = data['LEARNING_RATE']

    #Scheaduler settings
    STEP_SIZE       = data['STEP_SIZE']
    GAMMA           = data['GAMMA']

    #   Trainer settings
    EPOCHS          = data['EPOCHS']
    LOG_INTERVAL    = data['LOG_INTERVAL']
    SAVE_INTERVAL   = data['SAVE_INTERVAL']

    #Data objects
    dataset = Dataset(DATASET_PATH, CONTEXT_LEN)
    loader = Loader(dataset.get_dataset('train'), BATCH_SIZE, CONTEXT_LEN, NUM_EMBEDDINGS)

    #Model
    Leia = Transformer(NUM_EMBEDDINGS, CONTEXT_LEN, N_LAYERS, EMBEDDING_DIM, D_FF, N_HEADS, BIAS, P_DROP, DEVICE)
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