import os
import logging
import numpy as np
import pandas as pd
from functools import partial
from sklearn.model_selection import StratifiedKFold, KFold

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from config import get_args
from trainer import Trainer
from dice_loss import dice_loss
from utils import seed_everything
from models import Temp
from data import DataSet
from auto_batch_size import max_gpu_batch_size

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed) #fix seed
    device = torch.device('cuda:0') #use cuda:0

    if args.continue_train > 0:
        result_path = args.continue_from_folder
    else:
        result_path = os.path.join(args.result_path, args.pretrained_model.replace('/', '_') +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))    
    logger.info(args)
    #logger to log result of every output

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

    if args.continue_train > 0:
        prediction = pd.read_csv(os.path.join(result_path, 'sum.csv'))
        test_result = prediction[output_index].values
        stackking_input = pd.read_csv(os.path.join(result_path, f'for_stacking_input.csv'))

    skf = StratifiedKFold(n_splits=args.cv_k, random_state=args.seed, shuffle=True) #Using StratifiedKFold for cross-validation    
    for fold, (train_index, valid_index) in enumerate(skf.split(train_data['path'], train_data['label'])): #by skf every fold will have similar label distribution
        if args.continue_train > fold+1:
            logger.info(f'skipping {fold+1}-fold')
            continue
        fold_result_path = os.path.join(result_path, f'{fold+1}-fold')
        os.makedirs(fold_result_path)
        fold_logger = logger.getChild(f'{fold+1}-fold')
        fold_logger.handlers.clear()
        fold_logger.addHandler(logging.FileHandler(os.path.join(fold_result_path, 'log.log')))    
        fold_logger.info(f'start training of {fold+1}-fold')
        #logger to log current n-fold output

        kfold_train_data = train_data.iloc[train_index]
        kfold_valid_data = train_data.iloc[valid_index]

        train_dataset = DataSet(file_list=kfold_train_data['path'].values, label=kfold_train_data['label'].values)
        valid_dataset = DataSet(file_list=kfold_valid_data['path'].values, label=kfold_valid_data['label'].values)

        model = Temp(args).to(device) #make model based on the model name and args
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, #pin_memory=True
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, #pin_memory=True
        )
        
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs, after_scheduler=None)
        trainer = Trainer(
            train_loader, valid_loader, model, loss_fn, optimizer, scheduler_warmup, device, args.patience, args.epochs, fold_result_path, fold_logger, len(train_dataset), len(valid_dataset))
        trainer.train() #start training

        test_dataset = DataSet(file_list=test_data['path'].values, label=test_data['label'].values)
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        ) #make test data loader

        prediction[output_index] += trainer.test(test_loader) #softmax applied output; accumulate test prediction of current fold model
        prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False) 
        
        stackking_input.loc[valid_index, output_index] = trainer.test(valid_loader) #use the validation data(hold out dataset) to make input for Stacking Ensemble model(out of fold prediction)
        stackking_input.to_csv(os.path.join(result_path, f'for_stacking_input.csv'), index=False)

        '''np.savez_compressed(os.path.join(fold_result_path, 'test_prediction'), trainer.test(test_loader))
        np.savez_compressed(os.path.join(fold_result_path, 'valid_prediction'), trainer.test(valid_loader))
        np.savez(os.path.join(fold_result_path, 'valid_index'), valid_index)''' # case when output size is big

'''prediction['label'] = np.argmax(test_result, axis=-1) #use the most likely results as my final prediction
prediction.drop(columns=output_index).to_csv(os.path.join(result_path, 'prediction.csv'), index=False)''' #classification