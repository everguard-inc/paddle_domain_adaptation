# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import argparse
import numpy as np
import paddle
from typing import Dict
import yaml
from paddleseg.utils import logger, get_sys_env
import warnings
warnings.filterwarnings("ignore")
import utils
from cvlibs import Config
from script.train import Trainer
from datasets import Domen1Dataset, Domen2Dataset, get_augmentation

#paddle.disable_static()
#print('\n\n\n')
#print(paddle.in_dynamic_mode())  # True, Now we are in dynamic mode
#print('\n\n\n')

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # params of training
    parser.add_argument(
        "--config", dest="cfg", help="The config file.",
        default='configs/deeplabv2/custom.yml',
        type=str)
    parser.add_argument(
        '--iters',
        dest='iters',
        help='iters for training',
        type=int,
        default=400000)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=2)
    parser.add_argument(
        '--learning_rate',
        dest='learning_rate',
        help='Learning rate',
        type=float,
        default=None)
    parser.add_argument(
        '--save_interval',
        dest='save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--resume_model',
        dest='resume_model',
        help='The path of resume model',
        type=str,
        default=None)
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the model snapshot',
        type=str,
        default='./output')
    parser.add_argument(
        '--keep_checkpoint_max',
        dest='keep_checkpoint_max',
        help='Maximum number of checkpoints to save',
        type=int,
        default=5)
    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='Num workers for data loader',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        dest='do_eval',
        help='Eval while training',
        action='store_true')
    parser.add_argument(
        '--log_iters',
        dest='log_iters',
        help='Display logging information at every log_iters',
        default=10,
        type=int)
    parser.add_argument(
        '--use_vdl',
        dest='use_vdl',
        help='Whether to record the data to VisualDL during training',
        action='store_true')
    parser.add_argument(
        '--seed',
        dest='seed',
        help='Set the random seed during training.',
        default=42,
        type=int)
    parser.add_argument(
        '--fp16', dest='fp16', help='Whther to use amp', action='store_true')
    parser.add_argument(
        '--data_format',
        dest='data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')

    return parser.parse_args()

def load_yaml(path: str) -> Dict:
    with open(path, "r") as stream:
        try:
            content = yaml.load(stream, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return content

def main(args):
    #paddle.disable_static()
    #print(paddle.in_dynamic_mode())

    if args.seed is not None:
        paddle.seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        logger.info('Set seed to {}'.format(args.seed))

    env_info = get_sys_env()
    info = ['{}: {}'.format(k, v) for k, v in env_info.items()]
    info = '\n'.join(['', format('Environment Information', '-^48s')] + info +
                     ['-' * 48])
    logger.info(info)

    place = 'gpu' if env_info['Paddle compiled with cuda'] and env_info[
        'GPUs used'] else 'cpu'

    paddle.set_device(place)

    if not args.cfg:
        raise RuntimeError('No configuration file specified.')

    cfg = Config(
        args.cfg,
        learning_rate=args.learning_rate,
        iters=args.iters,
        batch_size=args.batch_size)
    
    cfg_data = load_yaml(args.cfg)['data']
    
    
    Domen1_images_train = [image for image in os.listdir(cfg_data['train_images_root']) if cfg_data['source_prefix'] in image]
    Domen2_images_train = [image for image in os.listdir(cfg_data['train_images_root']) if cfg_data['source_prefix'] not in image]
    Domen1_images_val = [image for image in os.listdir(cfg_data['val_images_root']) if cfg_data['source_prefix'] in image]
    Domen2_images_val = [image for image in os.listdir(cfg_data['val_images_root']) if cfg_data['source_prefix'] not in image]

    train_dataset_src = Domen1Dataset(
            Domen1_images_train, cfg_data['train_images_root'], 
            cfg_data['train_images_masks_root'], split='train', num_classes = cfg_data['num_classes'], 
            training = True , edge = False, resize =  cfg_data['size']
    )
    val_dataset_src = Domen1Dataset(
            Domen1_images_val, cfg_data['val_images_root'], 
            cfg_data['val_images_masks_root'], split='val', training = False, 
            edge = False, resize =  cfg_data['size']
    )
    train_dataset_tgt = Domen2Dataset(
            Domen2_images_train, cfg_data['train_images_root'], 
            cfg_data['train_images_masks_root'], split='train', training = True, 
            edge = False
    )
    val_dataset_tgt = Domen2Dataset(
            Domen2_images_val, cfg_data['val_images_root'], 
            cfg_data['val_images_masks_root'], split='val', training = False, 
            edge = False
    )

    val_dataset_tgt = val_dataset_tgt if args.do_eval else None
    val_dataset_src = val_dataset_src if args.do_eval else None

    if train_dataset_src is None:
        raise RuntimeError(
            'The training dataset is not specified in the configuration file.')
    elif len(train_dataset_src) == 0:
        raise ValueError(
            'The length of train_dataset is 0. Please check if your dataset is valid'
        )

    #msg = '\n---------------Config Information---------------\n'
    #msg += str(cfg)
    #msg += '------------------------------------------------'
    #logger.info(msg)

    trainer = Trainer(model=cfg.model, cfg=cfg.dic)
    trainer.train(
        train_dataset_src,
        train_dataset_tgt,
        val_dataset_tgt=val_dataset_tgt,
        val_dataset_src=val_dataset_src,
        optimizer=cfg.optimizer,
        save_dir=args.save_dir,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        resume_model=args.resume_model,
        save_interval=args.save_interval,
        log_iters=args.log_iters,
        num_workers=args.num_workers,
        use_vdl=args.use_vdl,
        keep_checkpoint_max=args.keep_checkpoint_max,
        test_config=cfg.test_config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
