#!/usr/bin/env python3
'''
7/26/2024: Making a script to train test models.
Choices needed for training:
- simulation to use for training data
- interparticle/intertracer separation for density field
- model hyperparameters such as:
    - number of layers
    - number of neurons per layer
    - loss function (SCCE, FOCAL, etc.)
    - learning rate
    - batch size
    - number of epochs
    - early stopping patience
    - LR scheduler/plateau patience

Definitions:
SIM: simulation to use for training data
PATH_TNG: path to TNG data
PATH_BOL: path to BOL data
PATH_DATA: either PATH_TNG or PATH_BOL depending on SIM
DEN_NAME: name of density file
FILE_DEN: PATH_DATA + DEN_NAME, the full path to the density file
MSK_NAME: name of mask file
FILE_MSK: PATH_DATA + MSK_NAME, the full path to the mask file

PATH_MODEL: path to save model
MODEL_NAME: name of model
FILE_MODEL: PATH_MODEL + MODEL_NAME, the full path to the model
(NOTE that training .csv log files are saved to PATH_MODEL as 
MODEL_NAME + '_log.csv')

PATH_FIGS: path to save figures
PATH_PREDS: path to save predictions, 
FILE_PRED: PATH_PREDS + MODEL_NAME + '_pred.fvol', the full path to the prediction

Input Args:

Output Products:
- trained model saved to PATH_MODEL
- training log saved to PATH_MODEL as MODEL_NAME + '_log.csv'
- figures saved to PATH_FIGS + SIM + MODEL_NAME + whatever fig name.png
- scores saved to PATH_MODEL appended to end of model_scores.csv
- predictions saved to PATH_PREDS as MODEL_NAME + '_pred.fvol'
'''
print('>>> Running gemini_test/train.py...')
import os
import sys
import nets
import argparse
import numpy as np
import absl.logging
import tensorflow as tf
from datetime import datetime
absl.logging.set_verbosity(absl.logging.ERROR)
print('TensorFlow version:', tf.__version__)
print('Keras version:', tf.keras.__version__)
print('Python version:', sys.version)
print('CUDA installed:', tf.test.is_built_with_cuda())
print('GPU available:', tf.test.is_gpu_available())
print('GPU device:', tf.test.gpu_device_name())
# set mixed precision policy NOTE has thrown errors before
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
#===============================================================================
# set random seeds for reproducibility
#===============================================================================
seed = 12; print('Random seed:', seed)
np.random.seed(seed); tf.random.set_seed(seed)
#===============================================================================
# set parameters and define paths
#===============================================================================
class_labels = ['Void', 'Wall', 'Filament', 'Halo']
N_CLASSES = len(class_labels)
PATH_TNG = '/ifs/groups/vogeleyGrp/data/TNG/'
PATH_BOL = '/ifs/groups/vogeleyGrp/data/Bolshoi/'
PATH_MODEL = '/ifs/groups/vogeleyGrp/models/_gemini_tests/'
PATH_FIGS = '/ifs/groups/vogeleyGrp/figs/FIGS_GEMINI/'
PATH_PREDS = '/ifs/groups/vogeleyGrp/preds/_gemini_tests/'
FILE_MODEL_SCORES = PATH_MODEL + 'model_scores.csv'
#===============================================================================
# parse command line arguments
#===============================================================================
parser = argparse.ArgumentParser(
    prog='train.py',
    description='Train a deepvoid model on TNG or Bolshoi data.')
required = parser.add_argument_group('required arguments')
required.add_argument(
    '-s', '--sim', type=str, required=True,
    help='Simulation to use for training data: TNG or BOL.')
required.add_argument(
    '-l', '--lamb', type=float, required=True,
    help='Interparticle/intertracer separation for density field.')
required.add_argument(
    '-d', '--depth', type=int, required=True,
    help='Depth of U-Net model.')
required.add_argument(
    '-f', '--filters', type=int, required=True,
    help='Number of filters in first layer of U-Net model.')
required.add_argument(
    '-b', '--batch', type=int, required=True,
    help='Batch size for training.')
required.add_argument(
    '-e', '--epochs', type=int, required=True,
    help='Number of epochs for training.')
required.add_argument(
    '-p', '--patience', type=int, required=True,
    help='Early stopping patience.')
required.add_argument(
    '-r', '--rate', type=float, required=True,
    help='Learning rate for training.')
required.add_argument(
    '-lf', '--loss', type=str, required=True,
    help='Loss function for training.')
required.add_argument(
    '-rot', '--rotation', type=bool, required=True,
    help='Which version of the density field to use.')
optional = parser.add_argument_group('optional arguments')
optional.add_argument(
    '-bn', '--batchnorm', type=bool, default=True,
    help='Batch normalization for U-Net model.')
optional.add_argument(
    '-do', '--dropout', type=float, default=0.0,
    help='Dropout rate for U-Net model.')
optional.add_argument(
    '-sche', '--scheduler', type=bool, default=False,
    help='Use learning rate scheduler.')
optional.add_argument(
    '-plt', '--plateau', type=bool, default=True,
    help='Use learning rate plateau.')
optional.add_argument(
    '-lm', '--load_model', type=str, default=False,
    help='Load a previously trained model.')
optional.add_argument(
    '-suff', '--suffix', type=str, default='',
    help='Suffix to append to model name.')
optional.add_argument(
    '-foc_alpha', '--focal_alpha', type=float, nargs='+',
    default=[0.25, 0.25, 0.25, 0.25],
    help='Alpha values for focal loss.')
optional.add_argument(
    '-foc_gamma', '--focal_gamma', type=float, default=2.0,
    help='Gamma value for focal loss.')
optional.add_argument(
    '-foc_norm', '--focal_norm', type=bool, default=False,
    help='Normalize focal loss.')
optional.add_argument(
    '-foc_clip', '--focal_clip', type=bool, default=False,
    help='Clip focal loss.')
optional.add_argument(
    '-foc_bal', '--focal_bal', type=bool, default=False,
    help='Balance focal loss.')
args = parser.parse_args()
print('Arguments:', args)
SIM = args.sim; L = args.lamb; D = args.depth; F = args.filters
BS = args.batch; ES = args.epochs; P = args.patience; LR = args.rate
LOSS = args.loss; ROT = args.rotation; BN = args.batchnorm
DO = args.dropout; SCHE = args.scheduler; PLT = args.plateau
LM = args.load_model; SUFF = args.suffix
FOC_ALPHA = args.focal_alpha; FOC_GAMMA = args.focal_gamma
FOC_NORM = args.focal_norm; FOC_CLIP = args.focal_clip
FOC_BAL = args.focal_bal
#===============================================================================
# initialize paths and filenames given SIM and L
#===============================================================================
BASE_FLAG = False; SUBGRID = 128; OFF = 64
LAMBDA_TH = 0.65
if L == 0.33 or L == 0.122:
    BASE_FLAG = True
if SIM == 'TNG' or SIM == 'TNG300':
    BoxSize = 205.0 # Mpc/h
    GRID = 512; SIG = 2.4
    PATH_DATA = PATH_TNG
    FILE_FIGS = PATH_FIGS + 'TNG/'
if SIM == 'BOL' or SIM == 'BOLSHOI' or SIM == 'Bolshoi':
    BoxSize = 256.0 # Mpc/h
    GRID = 640; SIG = 0.916
    PATH_DATA = PATH_BOL
    FILE_FIGS = PATH_FIGS + 'BOL/'
if BASE_FLAG:
    DEN_NAME = f'DM_density_GRID={GRID}_ROT={ROT}.fvol' # FULL DM DENSITY
else:
    DEN_NAME = f'halo_L={L}_GRID={GRID}_ROT={ROT}.fvol' # HALO DENSITY
MSK_NAME = f'mask_LTH={LAMBDA_TH}_SIG={SIG}_GRID={GRID}_ROT={ROT}.fvol'
FILE_DEN = PATH_DATA + DEN_NAME
FILE_MSK = PATH_DATA + MSK_NAME
if not os.path.exists(FILE_FIGS):
    os.makedirs(FILE_FIGS)
#===============================================================================
# define model name and paths
#===============================================================================
MODEL_NAME = f'{SIM}_D{D}_F{F}_L={L}_LOSS={LOSS}_GRID={GRID}_ROT={ROT}'
if BN:
    MODEL_NAME += '_BN'
if DO > 0.0:
    MODEL_NAME += f'_DO={DO}'
# add suffix to model name
if SUFF:
    MODEL_NAME += f'_{SUFF}'
FILE_MODEL = PATH_MODEL + MODEL_NAME
#===============================================================================
# load data
#===============================================================================
print('>>> Loading data...')
### Old way of loading data:
LOAD_INTO_MEMORY = False; print('Load all subcubes into mem?:', LOAD_INTO_MEMORY)
LOW_MEM = False; print('Use lower memory load of subcubes:', LOW_MEM)
if LOAD_INTO_MEMORY:
    print('Loading data into memory...')
    if LOW_MEM:
        features, labels = nets.load_dataset_all(FILE_DEN, FILE_MSK, SUBGRID)
    else:
        features, labels = nets.load_dataset_all_overlap(FILE_DEN, FILE_MSK,
                                                         SUBGRID, OFF)
    print('Data loaded into memory.')
    print('Features shape:', features.shape)
    print('Labels shape:', labels.shape)
    test_size = 0.2 # NOTE ADJUST VALIDATION SET SIZE HERE
    X_index = np.arange(features.shape[0])
    X_train, X_test, Y_train, Y_test = nets.train_test_split(X_index, labels,
                                                             test_size=test_size,
                                                             random_state=seed)
    X_train = features[X_train]; X_test = features[X_test]
    del features, labels # free up memory
    print(f'>>> Split into training ({(1-test_size)*100}%) and validation ({test_size*100}%) sets')
    print('Training features shape:', X_train.shape); print('Training labels shape:', Y_train.shape)
    print('Validation features shape:', X_test.shape); print('Validation labels shape:', Y_test.shape)
    if LOSS != 'SCCE':
        Y_train = nets.to_categorical(Y_train, N_CLASSES)
        Y_test  = nets.to_categorical(Y_test, N_CLASSES)
        print('Training labels shape (one-hot):', Y_train.shape)
        print('Validation labels shape (one-hot):', Y_test.shape)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    del X_train, Y_train # free up memory
else:
    print('Using data generator to load memmapped files...')
    
