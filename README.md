# GEMINI_TEST
Repository for tests and other misc scripts.

Files:
- train.py: training script
- pred.py: prediction script
- nets.py: helper functions script
- volumes.py: binary volumes reader script

Naming conventions:
GRID: number of voxels on a side of the cube. For TNG300-3, GRID=512, for Bolshoi GRID=640.
SUBGRID: number of voxels on a side of the subcube. So far, always SUBGRID=128.
OFF: offset of each subcube from the next. So far, always OFF=64.
ROT: rotation of the subcube. So far, always ROT=0 or 45.
L: lambda or interparticle or intertracer separation. For full DM density TNG L=0.33, for full Bolshoi L=0.122. Each halo/subhalo cube has its own L, so far we have L=1,3,5,7,10 for TNG and Bolshoi.
L_TH: lambda_th or threshold for the tidal mask. So far, always L_TH=0.65.
SIG: sigma or std dev of Gauss kernel used to smooth the tidal mask. For TNG, SIG=1.2, for Bolshoi SIG=2.4.

Full DM particle density cube: `DM_density_GRID=GRID_ROT=ROT.fvol`
Halo/subhalo cube: `halo_L=L_GRID=GRID_ROT=ROT.fvol`
Mask cube: `mask_LTH=LTH_SIG=SIG_GRID=GRID_ROT=ROT.fvol`

Validation data volumes:
X_train: `X_train_SIM=SIM_L=L_GRID=GRID_ROT=ROT.npy`
X_test: `X_test_SIM=SIM_L=L_GRID=GRID_ROT=ROT.npy`
Y_train: `Y_train_SIM=SIM_L=L_GRID=GRID_ROT=ROT.npy`
Y_test: `Y_test_SIM=SIM_L=L_GRID=GRID_ROT=ROT.npy`
(may not always be .npy files)

Model files:
`SIM_DDEPTH_FFILTERS_L=L_LOSS=LOSS_GRID=GRID_ROT=ROT`

To do in the future:
- Write up a markdown file for the full process of going from a simulation's particles to density cube and tidal mask, training models, scoring the predictions, and visualizing the predictions & scores.
- Graph neural network?
- GNN with real survey format data?
- Attention mechanisms?
- Additional data processing?
- Random crops for data augmentation
- Faster data loading (maybe using HDF5 files or separate scripts to load from disk faster)
- Residual blocks? Recurrent blocks? skip blocks? Inception networks?
- Dropout? L2 reg?
- 3D web visualization?
- Vision transformer?

