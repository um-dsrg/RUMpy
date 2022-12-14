Model Evaluation
================

Model evaluation is all carried out using the ```eval_sisr``` command.  This command can allow you to either simply super-resolve an input directory, or super-resolve the provided images and produce reports containing PSNR/SSIM/LPIPS results and model comparison images (examples provided below).

Models trained using the standard setup will produce a results folder containing the relevant configuration, training results and saved checkpoints.  When specifying a model for evaluation, the model base folder needs to be provided, and the specific epoch checkpoint to load.  The model handler will automatically take care of setting up the model and loading the checkpoint, if the correct configuration is provided (more details on specifying models below).

Model evaluation will always produce a folder with the specified results name, copy in its given config into this folder, and create an output folder for each model specified.
## All Configuration Options
As with training, options are specified using a ```.toml``` config file.  The below are all the accepted config parameters:

### Data Options
- ```--lr_dir``` - Input image directory.
- ```--hr_dir``` - Corresponding HR image directory (can leave empty if not computing metrics).
- ```--lr_dir_interp``` - Optional corresponding bicubically-interpolated LR images (can leave unspecified).
- ```--data_attributes``` - Location of image attributes (e.g. facial features file for CelebA images).
- ```--use_celeba_blacklist``` - Set this flag to remove any images which form part of the celeba blacklist.
- ```--dataset_name``` - Specify dataset name to use associated specific eval split.
- ```--data_split``` - Specifies data split to extract (train/test/eval).  Defaults to eval if not specified.
- ```--image_shortlist``` - Location of text file containing image names to select from input folder.
- ```--qpi_selection``` - Set these values to enforce qpi range when selecting test data.
- ```--full_directory``` - Set this flag to run analysis on all images provided.
- ```--standard_data``` - Set this flag to use one of the standard server locations (x4, x8, random_QPI).
- ```--use_test_group``` - Set this flag to run results only on typical 100 images.
- ```--metadata_file``` - Location of datafile containing metadata information. Defaults to degradation_metadata.csv if not specified.
- ```--id_source``` - Image ID file, when calculating face recognition metrics.
- ```--gallery_source``` - VGG Evaluation gallery directory (when using multiple reference galleries).
- ```--galleries``` - Specify galleries to use for face-recognition stats.
- ```--gallery_ref_images``` - Location of gallery reference images, if different from HR directory.
- ```--augmentation_normalization``` - Set to true to normalize all incoming metadata, or specify a list of the specific attributes that should be normalized.
- ```--ignore_degradation_location``` - Set to true to ignore degradation positioning information (e.g. 1-x).
### Processing/Output Options
- ```--out_loc``` - Output directory.
- ```--results_name``` - Output folder name.
- ```-m```, ```--metrics``` - List of metrics to calculate for each image (requires corresponding HR images to be provided).
- ```--save_im``` - Set to true to save all super-resolved images.
- ```--face_rec_profiling``` - Set this flag to evaluate face-recognition stats on given images.
- ```--batch_size``` - Batch size for parallel data loading.
- ```--model_only``` - Set this flag to skip all metrics and simply output super-resolved images.
- ```--no_image_comparison``` - Set this flag to disable image comparison generation.
- ```--save_raw_features``` - Set this flag to save raw features generated by face recognition network.
- ```--num_image_save``` - Set the maximum number of image comparisons to save.
- ```--save_data_model_folders``` - Set this flag to have FR metrics saved directly to the input model folders.
- ```--time_models/--no-time_models``` - Specify whether time model execution.  Defaults to on.
- ```--run_lpips_on_gpu``` - Set this flag to run LPIPS metrics on GPU.
- ```--lanczos_upsample``` - Set this flag to generate an additional LR image based on Lanczos upsampling.

### Model Options
- ```--model_loc``` - Model save location.
- ```-me```, ```--model_and_epoch``` - Models to evaluate.  Specified as ('Name', 'Epoch Selected').  Can specify multiple of these models (see examples).  Set epoch to 'best' to use epoch with best validation PSNR.
- ```--gpu/--no-gpu``` - Set to use GPU for evaluation (or leave blank to use CPU).
- ```--scale``` - Super-resolution scale.
- ```--hr_selection``` - Set the frame number to use as HR (GT) when 'data_type' is set to 'multi-frame'.
- ```--in_features``` - Set the number of features (channels) of the input image (==3 for RGB images).
- ```--use_mps``` - Set this flag to use MPS as GPU device.
## Examples
### Running a Config File
To run an evaluation routine with a specific config file, run ```eval_sisr --config path/to/config/file``` to run the evaluation.
### Super-Resolving images and evaluating them against their HR counterparts
Run the following config file (also provided as example_evaluation_config.toml):
```toml
metrics = ['PSNR', 'SSIM']  # metrics to calculate on validation set
model_and_epoch = [['rcan_cosine_v2', 'best'], ['q-rcan_cosine', 'best']]
results_name = 'simple_test'
gpu = true
save_im = true
scale = 4
model_loc = './Results/SISR/div_flickr/trained_models/standard_best/'
hr_dir = './Data/example_data/Set5/hr'
lr_dir = './Data/example_data/Set5/lr_random_blur'
batch_size = 1
out_loc = '/Users/matt/Desktop'
full_directory = true
recursive = true
```
### Super-Resolving all images without any comparison calculations
```toml
model_and_epoch = [['rcan_cosine_v2', 'best']]
model_only = true
results_name = 'simple_test_no_metrics'
gpu = true
save_im = true
scale = 4
model_loc = './Results/SISR/div_flickr/trained_models/standard_best/'
lr_dir = './Data/example_data/Set5/lr_random_blur'
batch_size = 1
out_loc = '/Users/matt/Desktop'
full_directory = true
recursive = true
```
