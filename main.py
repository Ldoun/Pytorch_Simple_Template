import os
import sys
import logging
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from models import Temp
from data import DataSet
from trainer import Trainer
from config import get_args
from lr_scheduler import get_sch
from utils import seed_everything, handle_unhandled_exception, save_to_json

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.model +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)

    train_data = pd.read_csv(args.train)
    train_data['path'] = train_data['path'].apply(lambda x: os.path.join(args.path, x))
    test_data = pd.read_csv(args.test)
    test_data['path'] = test_data['path'].apply(lambda x: os.path.join(args.path, x))
    #fix path based on the data dir

    input_size = None
    output_size = None

    prediction = pd.read_csv(args.submission)
    output_index = [f'{i}' for i in range(0, output_size)]
    stackking_input = pd.DataFrame(columns = output_index, index=range(len(train_data))) #dataframe for saving OOF predictions
  
    train_index, valid_index = train_test_split(range(len(train_data['path'])), test_size=args.test_size, shuffle=True, random_state=args.seed, stratify=train_data['label'])
    seed_everything(args.seed) #fix seed

    kfold_train_data = train_data.iloc[train_index]
    kfold_valid_data = train_data.iloc[valid_index]

    train_dataset = DataSet(file_list=kfold_train_data['path'].values, label=kfold_train_data['label'].values)
    valid_dataset = DataSet(file_list=kfold_valid_data['path'].values, label=kfold_valid_data['label'].values)

    model = Temp(args).to(device) #make model based on the model name and args
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_sch(args.scheduler, optimizer, warmup_epochs=args.warmup_epochs, epochs=args.epochs)

    if args.continue_train_from is not None:
        state = torch.load(args.continue_train_from)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        epoch = state['epoch']
    else:
        epoch = 0

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, #pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, #pin_memory=True
    )
    
    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, args.patience, args.epochs, result_path, logger, start_epoch=epoch)
    trainer.train() #start training

    test_dataset = DataSet(file_list=test_data['path'].values, label=test_data['label'].values)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    ) #make test data loader

    prediction[output_index] += trainer.test(test_loader) #softmax applied output; accumulate test prediction of current fold model
    prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False) 