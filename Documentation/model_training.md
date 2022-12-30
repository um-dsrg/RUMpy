Model Training (SISR and Regression)
==============
Training a model requires 2 steps:
- Configuration file preparation.
- Model training code execution.

Model training will produce a folder at the specified directory, within which all results and a copy of the provided configuration parameters are placed.

Preparing Configuration Files
----------------------------
Configuration files (in [```.toml```](https://github.com/toml-lang/toml) format) can be used to specify all options for training, file saving and any internal parameters for the selected model.  Examples of config files can be found under ```rumpy/SISR/args/```.

### Training Options

Training options should be specified under the ```[training]``` tag.  These include:
- ```num_epochs``` - Specifies the number of epochs to run for training. 
- ```continue_from_epoch``` - Specify the epoch from which to continue training.  Assumes the selected model starting point is saved in the specified experiment folder.
- ```metrics``` - Metrics to evaluate on validation set during training (currently supports PSNR, SSIM and LPIPS).
- ```seed``` -  Random seed for ensuring reproducible model initializations.
- ```early_stopping_patience``` - This parameter specifies the number of epochs of continuous validation loss increase to wait before stopping training.  Not specifying this parameter assumes no early stopping is required.
- ```overwrite_data``` - Specify this flag if new models can overwrite any saved data already present.
- ```branch_root``` - Specifies the branch from which to continue training.
- ```new_branch``` - Specify this flag to start a new folder for saving results.
- ```new_branch_name``` - Name of new branch, if required.
- ```logging``` - Set to 'visual' to activate loss function graphical logging during training.
- ```save_samples``` - Set this flag to save some image samples (from the validation set) after each epoch.
- ```gpu``` - Set to 'single' to use one gpu, 'multiple' to use all available gpus or 'off' to not use any GPUs (default off).
- ```sp_gpu``` - Specify gpu ID to use (if in single mode).
- ```aim_track``` - Set to true to also log results using Aim (default True).
- ```aim_home``` -  Sets Aim output directory.  Defaults to Results/SISR.

### Model-Specific Options
Specify the model name under the ```[model]``` tag. Internal parameters should then be placed under the ```[model.internal_params]``` tag. 

Some general parameters:

- ```lr``` - Learning rate.
- ```scale``` - Super-resolution scale.
- ```scheduler``` - Type of learning rate scheduler to use (leave empty for no scheduler).

Some examples of model-specific parameters:
- EDSR
  - ```num_features``` - Number of channels to use per block (default 64).
  - ```num_blocks``` - Number of residual blocks in network (default 16).
  - ```res_scale``` - Residual scaling to use per block (default 0.1).
- SFTMD
  - ```metadata``` - Metadata to insert into model.
  - ```num_features``` - Number of channels to use per block (default 64).
  - ```num_blocks``` - Number of residual blocks in network (default 16).
  - ```SFT_type``` - Type of SFT block to use throughout network.

### Data Options
All data parameters are placed under the ```[data]``` tag.  General options include:

- ```batch_size``` - Batch size for training images.
- ```eval_batch_size``` - Batch size for validation images.
- ```dataloader_threads``` - Number of parallel threads to use for data handling.

Both training and validation datasets needs to be prepared as dictated in ```data_prep.md```.  Once prepared, a new ```[data.training_sets.data_x]``` needs to be specified per image folder.  Within each data tag, the following parameters can be specified:
- ```name``` - image dataset name (if within standard list, can use pre-specified train/eval split).  If this is not specified, the whole dataset will be used.
- ```lr``` - low-res image directory.
- ```lr_interp``` = Pre-scaled low-res images (using bicubic interpolation) - optional.
- ```hr``` = Corresponding high-res mage directory.
- ```qpi_values``` - Location of linked metadata (WIP).
- ```metadata``` - Specify a list of metadata which should be extracted and provided alongside the image data.
- ```crop``` - Specify this value if cropped patches should be extracted from each input image.
- ```random_augment``` - Set this flag to include random perturbations in dataset during training (flips and rotations).
- ```cutoff``` - Specify this value as the maximum amount of images to gather from the dataset.  Leave empty to place no restrictions.  
Each training run should have at least one ```training_set``` and at least one ```eval_set```.

Running Models
--------------

### General Config
Once all config parameters are set, models can be run using the command:

```train_sisr --parameters path/to/config/file``` 

Some additional parameters can be passed to ```train_sisr``` which will override those specified in the config file (check ```Code/net_train.py``` for examples).

During a training run, the system will display a progress bar for epochs run, a running tally of train/eval PSNR/loss and a graphical output of key metrics.  These are all placed within the ```result_outputs``` folder in the experiment main folder.  A model is saved at the end of each epoch, and is stored in ```saved_models``` in the base folder.

An example config file for training RCAN on DIV2K and Flickr2K is provided in this folder - example_training_config.toml.  Check this file for detailed comments on the selection of parameters for this training run.

### Using Aim

If using Aim, all metrics will also be logged to the selected Aim output folder, along with a number of system metrics (CPU %, GPU memory etc.).  To open these results, navigate to the aim output directory and run:

```aim up```

Details on how to navigate the Aim interface are provided [here](https://github.com/aimhubio/aim).
