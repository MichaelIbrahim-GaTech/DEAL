import argparse
import os
import time
from tkinter import E

import torch
from pytorch_lightning import seed_everything

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from log import logger
from model.ner_model import NERBaseAnnotator
from utils.reader import CoNLLReader

conll_iob = {"B-Archive": 0, "B-CelestialObject": 1, "B-CelestialObjectRegion": 2, "B-CelestialRegion": 3, "B-Citation": 4, "B-Collaboration": 5, "B-ComputingFacility": 6, "B-Database": 7, "B-Dataset": 8, "B-EntityOfFutureInterest": 9, "B-Event": 10, "B-Fellowship": 11, "B-Formula": 12, "B-Grant": 13, "B-Identifier": 14, "B-Instrument": 15, "B-Location": 16, "B-Mission": 17, "B-Model": 18, "B-ObservationalTechniques": 19, "B-Observatory": 20, "B-Organization": 21, "B-Person": 22, "B-Proposal": 23, "B-Software": 24, "B-Survey": 25, "B-Tag": 26, "B-Telescope": 27, "B-TextGarbage": 28, "B-URL": 29, "B-Wavelength": 30, "I-Archive": 31, "I-CelestialObject": 32, "I-CelestialObjectRegion": 33, "I-CelestialRegion": 34, "I-Citation": 35, "I-Collaboration": 36, "I-ComputingFacility": 37, "I-Database": 38, "I-Dataset": 39, "I-EntityOfFutureInterest": 40, "I-Event": 41, "I-Fellowship": 42, "I-Formula": 43, "I-Grant": 44, "I-Identifier": 45, "I-Instrument": 46, "I-Location": 47, "I-Mission": 48, "I-Model": 49, "I-ObservationalTechniques": 50, "I-Observatory": 51, "I-Organization": 52, "I-Person": 53, "I-Proposal": 54, "I-Software": 55, "I-Survey": 56, "I-Tag": 57, "I-Telescope": 58, "I-TextGarbage": 59, "I-URL": 60, "I-Wavelength": 61, "O": 62}

def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

    p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
    p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='conll')

    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=1500)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=100)

    p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--model', type=str, help='Model path.', default=None)
    p.add_argument('--model_name', type=str, help='Model name.', default=None)
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

    p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    p.add_argument('--accum_grad_batches', type=int, help='Number of batches for accumulating gradients.', default=1)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--cuda', type=str, help='Cuda Device', default='cuda:0')
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)

    return p.parse_args()


def get_tagset(tagging_scheme):
    if 'conll' in tagging_scheme:
        return conll_iob
    else:
        # If you choose to use a different tagging scheme, you may need to do some post-processing
        raise Exception("ERROR: Only conll tagging scheme is accepted")


def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[:model_name.rfind('.')]
    return '{}/{}_base_{}.tsv'.format(out_dir, prefix, model_name)


def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    logger.info('Finished writing evaluation performance for {}'.format(out_file))


def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large'):
    if file_path is None:
        return None
    reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model)
    reader.read_data(file_path)

    return reader


def create_model(train_data, dev_data, tag_to_id, batch_size=64, dropout_rate=0.1, stage='fit', lr=1e-5, encoder_model='xlm-roberta-large', num_gpus=1):
    return NERBaseAnnotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model,
                            dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)


def load_model(model_file, tag_to_id=None, stage='test'):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    model = NERBaseAnnotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    model.stage = stage
    return model, model_file


def save_model(trainer, out_dir, model_name, timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info('Stored model {}.'.format(outfile))
    return outfile


def train_model(model, out_dir='', epochs=10, gpus=1, model_name='', timestamp='', grad_accum=1):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs, model_name=model_name, timestamp=timestamp, grad_accum=grad_accum)
    trainer.fit(model)
    return trainer


def get_modelcheckpoint_callback(out_dir, model_name, timestamp):
    if not os.path.exists(out_dir + '/lightning_logs/'):
        os.makedirs(out_dir + '/lightning_logs/')

    if len(os.listdir(out_dir + '/lightning_logs/')) < 1:
        bcp_path  = out_dir + '/lightning_logs/version_0/checkpoints/'
    else:
        bcp_path  = out_dir + '/lightning_logs/version_' +  str(int(os.listdir(out_dir + '/lightning_logs/')[-1].split('_')[-1]) + 1) + '/checkpoints/'
       
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=bcp_path,
        filename=model_name + '_timestamp_' + str(timestamp) + '_final'
    )
    return checkpoint_callback


def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10, model_name='', timestamp='', grad_accum=1):
    seed_everything(42)
    logger = pl.loggers.CSVLogger(out_dir, name="lightning_logs")
    if is_test:
        return pl.Trainer(gpus=1, logger=logger) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)

    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=gpus, logger=logger, deterministic=True, max_epochs=epochs, callbacks=[get_model_earlystopping_callback(),get_modelcheckpoint_callback(out_dir, model_name, timestamp)],
                             default_root_dir=out_dir, distributed_backend='ddp', checkpoint_callback=True, accumulate_grad_batches=grad_accum)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=epochs, logger=logger, default_root_dir=out_dir)

    return trainer


def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor


def get_model_earlystopping_callback():
    es_clb = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        verbose=True,
        mode='min'
    )
    return es_clb


def get_models_for_evaluation(path):
    if 'checkpoints' not in path:
        path = path + '/checkpoints/'
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith('final.ckpt')]

    return models[0] if len(models) != 0 else None


def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files
