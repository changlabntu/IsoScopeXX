from __future__ import print_function
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import os, shutil, time, sys
from tqdm import tqdm
from dotenv import load_dotenv
from utils.make_config import load_json, save_json
import json
import yaml
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from dataloader.data_multi import PairedImageDataset as Dataset
from utils.get_args import get_args

os.environ['OPENBLAS_NUM_THREADS'] = '1'


def prepare_log(args):
    """
    finalize arguments, creat a folder for logging, save argument in json
    """
    args.not_tracking_hparams = []  # 'mode', 'port', 'epoch_load', 'legacy', 'threads', 'test_batch_size']
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/', exist_ok=True)
    os.makedirs(os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/', exist_ok=True)
    save_json(args, os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + '0.json')
    shutil.copy('models/' + args.models + '.py',
                os.environ.get('LOGS') + args.dataset + '/' + args.prj + '/' + args.models + '.py')
    return args


if __name__ == '__main__':
    parser = get_args()

    # 1) Do a preliminary parse to get --yaml and any CLI overrides
    prelim_args = parser.parse_known_args()[0]

    # 2) Load YAML config early to obtain defaults (including models)
    with open('env/' + prelim_args.yaml + '.yaml', 'rt') as f:
        json_args = argparse.Namespace()
        json_args.__dict__.update(yaml.safe_load(f)['train'])

    # 3) Resolve model name: CLI --models overrides YAML; otherwise use YAML
    models_name = getattr(prelim_args, 'models', None) or getattr(json_args, 'models', None)
    if models_name is None:
        raise ValueError("Model name not specified. Provide --models on the CLI or set 'models' in the YAML under 'train'.")

    # 4) Import model and augment parser with model-specific args
    GAN = getattr(__import__('models.' + models_name), models_name).GAN
    parser = GAN.add_model_specific_args(parser)

    # 5) Final parse with YAML defaults as the namespace, allowing CLI to override
    args = parser.parse_args(namespace=json_args)

    # environment file
    with open('env/env', 'r') as f:
        configs = json.load(f)[args.env]

    os.environ.setdefault('DATASET', configs['DATASET'])
    os.environ.setdefault('LOGS', configs['LOGS'])

    if args.env is not None:
        load_dotenv('env/.' + args.env)
    else:
        load_dotenv('env/.t09')

    # Finalize Arguments and create files for logging
    args.bash = ' '.join(sys.argv)
    args = prepare_log(args)
    print(args)

    # Load Dataset and DataLoader
    train_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/train/',
                        path=args.direction,
                        config=args, mode='train')#, index=None, filenames=True)

    train_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, drop_last=True)

    try:
        eval_set = Dataset(root=os.environ.get('DATASET') + args.dataset + '/val/',
                           path=args.direction,
                           config=args, mode='test')#, index=None, filenames=True)
        eval_loader = DataLoader(dataset=eval_set, num_workers=1, batch_size=args.test_batch_size, shuffle=False,
                                 pin_memory=True)
    except:
        eval_loader = None
        print('No validation set')

    # preload
    if args.preload:
        tini = time.time()
        print('Preloading...')
        for i, x in enumerate(tqdm(train_loader)):
            pass
        if eval_loader is not None:
            for i, x in enumerate(tqdm(eval_loader)):
                pass
        print('Preloading time: ' + str(time.time() - tini))

    # Logger
    os.makedirs(os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'logs'), exist_ok=True)
    logger = pl_loggers.TensorBoardLogger(os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'logs'))

    # Trainer
    checkpoints = os.path.join(os.environ.get('LOGS'), args.dataset, args.prj, 'checkpoints')
    os.makedirs(checkpoints, exist_ok=True)

    net = GAN(hparams=args, train_loader=train_loader, eval_loader=eval_loader, checkpoints=checkpoints)

    "Please use `Trainer(accelerator='gpu', devices=-1)` instead."
    trainer = pl.Trainer(gpus=-1, strategy='ddp_spawn',
                         max_epochs=args.n_epochs,  # progress_bar_refresh_rate=20,
                         logger=logger,
                         enable_checkpointing=True, log_every_n_steps=100,
                         check_val_every_n_epoch=1, accumulate_grad_batches=2)
    if eval_loader is not None:
        trainer.fit(net, train_loader, eval_loader)
    else:
        trainer.fit(net, train_loader)