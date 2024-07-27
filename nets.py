#!/usr/bin/env python3
'''
7/26/2024: Making a script to contain helper functions for training,
validation, and testing models. Also contains plotting functions.
'''
import re
import gc
import os
import sys
import volumes
import numpy as np
import tensorflow as tf
from scipy import ndimage as ndi
from keras.models import Model, load_model, clone_model
from sklearn.model_selection import train_test_split

class_labels = ['Void','Wall','Filament','Halo']
SUBGRID = 128; OFF = 64
#---------------------------------------------------------
# Summary statistics for a Numpy array
#---------------------------------------------------------
def summary(array):
  print('### Summary Statistics ###')
  print('Shape: ',str(array.shape))
  print('Mean: ',np.mean(array))
  print('Median: ',np.median(array))
  print('Maximum: ',np.max(array))
  print('Minimum: ',np.min(array))
  print('Std deviation: ',np.std(array))
  print('Variance: ',np.var(array))
#---------------------------------------------------------
# Regularization options: minmax and standardize
#---------------------------------------------------------
def minmax(a):
    return (a-np.min(a))/(np.max(a)-np.min(a))
def standardize(a):
  return (a-np.mean(a))/(np.std(a))
#---------------------------------------------------------
# Assemble cube from subcubes
#---------------------------------------------------------
def assemble_cube2(Y_pred,GRID,SUBGRID,OFF):
    cube  = np.zeros(shape=(GRID,GRID,GRID))
    #nbins = (GRID // SUBGRID) + 1 + 1 + 1
    nbins = (GRID // SUBGRID) + (GRID // SUBGRID - 1)
    #if GRID == 640:
    #  nbins += 1
    cont  = 0
    
    SUBGRID_4 = SUBGRID//4
    SUBGRID_2 = SUBGRID//2
    
    for i in range(nbins):
        if i==0:
            di_0 = SUBGRID*i - OFF*i
            di_1 = SUBGRID*i - OFF*i + SUBGRID_4+SUBGRID_2
            si_0 =  0
            si_1 = -SUBGRID_4
        else:
            di_0 = SUBGRID*i - OFF*i + SUBGRID_4
            di_1 = SUBGRID*i - OFF*i + SUBGRID_4+SUBGRID_2
            si_0 =  SUBGRID_4
            si_1 = -SUBGRID_4            
            if i==nbins-1:
                di_0 = SUBGRID*i - OFF*i + SUBGRID_4
                di_1 = SUBGRID*i - OFF*i + SUBGRID
                si_0 =  SUBGRID_4
                si_1 =  SUBGRID

        for j in range(nbins):
            if j==0:
                dj_0 = SUBGRID*j - OFF*j
                dj_1 = SUBGRID*j - OFF*j + SUBGRID_4+SUBGRID_2
                sj_0 =  0
                sj_1 = -SUBGRID_4
            else:
                dj_0 = SUBGRID*j - OFF*j + SUBGRID_4
                dj_1 = SUBGRID*j - OFF*j + SUBGRID_4+SUBGRID_2
                sj_0 = SUBGRID_4
                sj_1 = -SUBGRID_4
                if j==nbins-1:
                    dj_0 = SUBGRID*j - OFF*j + SUBGRID_4
                    dj_1 = SUBGRID*j - OFF*j + SUBGRID
                    sj_0 = SUBGRID_4
                    sj_1 = SUBGRID                     
            for k in range(nbins):
                if k==0:
                    dk_0 = SUBGRID*k - OFF*k
                    dk_1 = SUBGRID*k - OFF*k + SUBGRID_4+SUBGRID_2
                    sk_0 =  0
                    sk_1 = -SUBGRID_4
                else:
                    dk_0 = SUBGRID*k - OFF*k + SUBGRID_4
                    dk_1 = SUBGRID*k - OFF*k + SUBGRID_4+SUBGRID_2
                    sk_0 =  SUBGRID_4
                    sk_1 = -SUBGRID_4
                    if k==nbins-1:
                        dk_0 = SUBGRID*k - OFF*k + SUBGRID_4
                        dk_1 = SUBGRID*k - OFF*k + SUBGRID
                        sk_0 = SUBGRID_4
                        sk_1 = SUBGRID                                                                                                        
                    
                cube[di_0:di_1, dj_0:dj_1, dk_0:dk_1] = Y_pred[cont, si_0:si_1, sj_0:sj_1, sk_0:sk_1,0]
                cont = cont+1
    return cube
#---------------------------------------------------------
# For loading training and testing data for training
# if loading data for regression, ensure classification=False!!
#---------------------------------------------------------
def load_dataset_all(FILE_DEN, FILE_MASK, SUBGRID, preproc='mm', classification=True, sigma=None, binary_mask=False):
  '''
  Function that loads the density and mask files, splits into subcubes of size
  SUBGRID, rotates by 90 degrees three times, and returns the X and Y data.
  FILE_DEN: str filepath to density field.
  FILE_MASK: str filepath to mask field.
  SUBGRID: int size of subcubes.
  preproc: str preprocessing method. 'mm' for minmax, 'std' for standardize.
  classification: bool whether or not you're doing classification. def True.
  sigma: float sigma for Gaussian smoothing. def None.
  binary_mask: bool whether or not to convert mask to binary. def False. 
  '''
  print(f'Reading volume: {FILE_DEN}... ')
  den = volumes.read_fvolume(FILE_DEN)
  if sigma is not None:
    den = ndi.gaussian_filter(den,sigma,mode='wrap')
    print(f'Smoothed density with a Gaussian kernel of size {sigma}')
  print(f'Reading mask: {FILE_MASK}...')
  msk = volumes.read_fvolume(FILE_MASK)
  # print mask populations:
  _, cnts = np.unique(msk,return_counts=True)
  for val in cnts:
    print(f'% of population: {val/den.shape[0]**3 * 100}')
  den_shp = den.shape
  #msk_shp = msk.shape
  summary(den); summary(msk)
  # binary mask oneliner
  if binary_mask == True:
    msk = (msk < 1.).astype(int)
    print('Converted mask to binary mask. Void = 1, not void = 0.')
    summary(den); summary(msk)
  if preproc == 'mm':
    #den = minmax(np.log10(den)) # this can create NaNs be careful
    den = minmax(den)
    #msk = minmax(msk) # 12/5 needed to disable this for sparse CCE losses
    print('Ran preprocessing to scale density to [0,1]!')
    print('\nNew summary statistics: ')
    summary(den)
  if preproc == 'std':
    den = standardize(den)
    #msk = standardize(msk)
    print('Ran preprocessing by dividing density/mask by std dev and subtracting by the mean! ')
    print('\nNew summary statistics: ')
    summary(den)
  # Make wall mask
  #msk = np.zeros(den_shp,dtype=np.uint8)
  n_bins = den_shp[0] // SUBGRID

  cont = 0 
  X_all = np.zeros(shape=((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1))
  if classification == False:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.float16)
  else:
    Y_all = np.ndarray(((n_bins**3)*4, SUBGRID,SUBGRID,SUBGRID,1),dtype=np.int8)

  for i in range(n_bins):
    for j in range(n_bins):
      for k in range(n_bins):
        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,2)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,2)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,1)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,1)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1

        sub_den = den[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_den = volumes.rotate_cube(sub_den,0)
        X_all[cont,:,:,:,0] = sub_den
        sub_msk = msk[i*SUBGRID:(i+1)*SUBGRID, j*SUBGRID:(j+1)*SUBGRID, k*SUBGRID:(k+1)*SUBGRID]
        sub_msk = volumes.rotate_cube(sub_msk,0)
        Y_all[cont,:,:,:,0] = sub_msk
        cont = cont+1
    #print(i,j,k)
  X_all = X_all.astype('float32')
  Y_all = Y_all.astype('int8')
  gc.collect()
  return X_all, Y_all
#---------------------------------------------------------
# 4/30/24: adding load_dataset_all_overlap function
# this function will ACTUALLY do what we claim
# load_dataset has done all along, which is taking overlapping
# subcubes, rotating by 90 degrees 3 times, for 
# a total of N_subcubes = 4 * [(GRID/SUBGRID) + (GRID/SUBGRID - 1)]^3
# NOTE THAT THIS IS FOR TRAINING ONLY!!!
#---------------------------------------------------------
def load_dataset_all_overlap(FILE_DEN, FILE_MSK, SUBGRID, OFF, preproc='mm', sigma=None):
  '''
  Function that loads density and mask files, splits into overlapping subcubes.
  Subcubes overlap by OFF, and are of size SUBGRID.
  Each subcube is rotated by 90 deg three times.
  FILE_DEN: str filepath to density field.
  FILE_MSK: str filepath to mask field.
  SUBGRID: int size of subcubes.
  OFF: int overlap of subcubes.
  preproc: str preprocessing method. 'mm' for minmax, 'std' for standardize.
  sigma: float sigma for Gaussian smoothing. def None.
  '''
  print(f'Reading volume: {FILE_DEN}... ')
  den = volumes.read_fvolume(FILE_DEN)
  if sigma is not None:
    den = ndi.gaussian_filter(den,sigma,mode='wrap')
    print(f'Smoothed density with a Gaussian kernel of size {sigma}')
  print(f'Reading mask: {FILE_MSK}...')
  msk = volumes.read_fvolume(FILE_MSK)
  # print mask populations:
  _, cnts = np.unique(msk,return_counts=True)
  for val in cnts:
    print(f'% of population: {val/den.shape[0]**3 * 100}')
  summary(den); summary(msk)
  if preproc == 'mm':
    den = minmax(den)
    print('Ran preprocessing to scale density to [0,1]!')
    print('\nNew summary statistics for density field: ')
    summary(den)
  if preproc == 'std':
    den = standardize(den)
    print('Ran preprocessing by dividing density/mask by std dev and subtracting by the mean! ')
    print('\nNew summary statistics for density field: ')
    summary(den)
  nbins = den.shape[0]//SUBGRID + (den.shape[0]//SUBGRID - 1)
  print(f'Number of overlapping subcubes: {4*nbins**3}')
  X_all_overlap = np.ndarray(((nbins**3)*4, SUBGRID, SUBGRID, SUBGRID, 1))
  Y_all_overlap = np.ndarray(((nbins**3)*4, SUBGRID, SUBGRID, SUBGRID, 1))
  # loop over overlapping subcubes, rotate!
  cont = 0
  for i in range(nbins):
    off_i = SUBGRID*i - OFF*i
    for j in range(nbins):
      off_j = SUBGRID*j - OFF*j
      for k in range(nbins):
        off_k = SUBGRID*k - OFF*k
        # define subcube:
        sub_den = den[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        sub_msk = msk[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
        # rot 90:
        sub_den = np.rot90(sub_den)
        sub_msk = np.rot90(sub_msk)
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
        # rot 180
        sub_den = np.rot90(sub_den)
        sub_msk = np.rot90(sub_msk)
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
        # rot 270
        sub_den = np.rot90(sub_den)
        sub_msk = np.rot90(sub_msk)
        X_all_overlap[cont,:,:,:,0] = sub_den
        Y_all_overlap[cont,:,:,:,0] = sub_msk
        cont += 1
  gc.collect()
  return X_all_overlap.astype('float32'), Y_all_overlap.astype('int8')

#---------------------------------------------------------
# For loading testing/validation data for prediction
#---------------------------------------------------------
def load_dataset(file_in, SUBGRID, OFF, preproc='mm',sigma=None,return_int=False):
  #--- Read density field
  den = volumes.read_fvolume(file_in)
  if sigma is not None:
    den = ndi.gaussian_filter(den,sigma,mode='wrap')
    print(f'Density was smoothed w/ a Gaussian kernel of size {sigma}')
  if preproc == 'mm':
    #den = minmax(np.log10(den)) # MUST MATCH PREPROC METHOD USED IN TRAIN
    den = minmax(den); print('Ran preprocessing to scale density to [0,1]!')
  if preproc == 'std':
    den = standardize(den); print('Ran preprocessing to scale density s.t. mean=0 and std dev = 1!')
  if preproc == None:
    pass
  #nbins = (den.shape[0] // SUBGRID) + 1 + 1 + 1 # hacky way
  #if den.shape[0] == 640:
  #  nbins += 1
  nbins = den.shape[0]//SUBGRID + (den.shape[0]//SUBGRID - 1)
  X_all = np.zeros(shape=(nbins**3, SUBGRID,SUBGRID,SUBGRID,1))
  
  cont  = 0
  for i in range(nbins):
    off_i = SUBGRID*i - OFF*i
    for j in range(nbins):
      off_j = SUBGRID*j - OFF*j
      for k in range(nbins):
        off_k = SUBGRID*k - OFF*k
        #print(i,j,k,'|', off_i,':',off_i+SUBGRID,',',off_j,':',off_j+SUBGRID,',',off_k,':',off_k+SUBGRID)
        sub_den = den[off_i:off_i+SUBGRID,off_j:off_j+SUBGRID,off_k:off_k+SUBGRID]
        X_all[cont,:,:,:,0] = sub_den
        cont = cont+1
      
  if return_int:
    X_all = X_all.astype('uint8')
  else:
    X_all = X_all.astype('float16')
  gc.collect()
  return X_all

#---------------------------------------------------------
# Data generator for training using memmapped npy files
#---------------------------------------------------------
def data_gen_mmap(FILE_DEN, FILE_MSK, BATCH_SIZE, SUBGRID_SIZE=128):
    '''
    Function to generate batches of data from memmapped npy files.
    Inputs:
    FILE_DEN: str filepath to density field.
    FILE_MSK: str filepath to mask field.
    BATCH_SIZE: int batch size.
    '''
    den = np.memmap(FILE_DEN, dtype='float32', mode='r',
                   shape=(None, SUBGRID_SIZE, SUBGRID_SIZE, SUBGRID_SIZE, 1))
    msk = np.memmap(FILE_MSK, dtype='int8', mode='r',
                   shape=(None, SUBGRID_SIZE, SUBGRID_SIZE, SUBGRID_SIZE, 1))
    N_subcubes = den.shape[0]
    while True:
      for i in range(0, N_subcubes, BATCH_SIZE):
        end = min(i+BATCH_SIZE, N_subcubes)
        # NOTE can add data augmentation here?
        yield den[i:end], msk[i:end]