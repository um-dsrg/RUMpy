SISR Data Preparation
======================

## Details
SISR data preparation can be mostly facilitated through the ```image_manipulate``` CLI function.  This function systematically applies a given image degradation pipeline to all images within a provided folder.  All possible parameters for this function are provided below (all details and defaults are also provided when running ```image_manipulate --help```):
### General Parameters
- ```source_dir``` - Input directory from which to source images.
- ```output_dir``` - Output directory within which to save new images.  Final directory name does not need to be created beforehand.
- ```pipeline``` - Pipeline of operations to perform, separated by "-".  Available operations include:
  - **jmcompress** - JM H.264 compression.  Requires JM to be installed (see main documentation for details).
  - **jpegcompress** - JPEG compression.
  - **randomcompress** - Randomly selects between JM H.264 and JPEG compression for each image provided.
  - **ffmpegcompress** - FFMPEG compression.
  - **downsample** - Bicubic downscaling.
  - **upsample** - Bicubic upsampling.
  - **srmdgaussianblur** - Gaussian blurring using SRMD's system.
  - **realesrganblur** - Blurring using Real-ESRGAN's system.
  - **bsrganblur** - Blurring using BSRGAN's system.
  - **realesrgannoise** - Noise addition using Real-ESRGAN's system.
- ```seed``` - Random seed used to initialize libraries.  Default 8.
- ```scale``` - Scale to use when downsampling.  Default 4.
- ```recursive``` - Set this flag to signal data converter to seek out all images in all sub-directories of main directory specified.
- ```output_extension``` - Final output image extension. 
- ```multiples``` - Number of degraded images to produce per input image.
- ```pipeline_config``` - Path to configuration file (TOML format) containing specific degradation parameters (more details below).
- ```override_cli``` - Fully override all default parameters using config file inputs.

### Config File Parameters
Specifying a degradation pipeline at the CLI will use all default parameters for each operation (typically random application within a pre-defined range).  To fine-tune and control each operation, a configuration file needs to be specified.  This configuration file can be supplied with all the parameters defined above.  However, defining degradation pipelines and parameters is more explicit:
#### Defining Config File Pipeline
To define a pipeline, each degradation must be supplied in list format along with a keyword for each individual operation e.g. to define a blurring + downsampling pipeline:

```toml
pipeline = [ [ "realesrganblur", "b-config",], [ "downsample", "d-config",],]
```
Here the first blurring operation has the unique ID ```b-config```.  With the pipeline defined, the parameters of each unique degradation ID can be specified in separate folder headings e.g.:

```toml
[deg_configs.b-config] # configuration for the b-config blur operation (must have the root heading deg_configs)
device = 0 # specific GPU to use for computation
pca_batch_len = 10000  # number of kernels to generate to compute PCA encoder
kernel_range = [ "iso", "aniso", "generalized_iso", "generalized_aniso", "plateau_aniso", "plateau_iso", "sinc",]  # possible kernel types to sample
pca_length = 100  # PCA elements for each kernel
request_full_kernels = false  # the full kernels should not be output to the metadata file
request_pca_kernels = false  # the PCA kernels are not required
request_kernel_metadata = true  # the kernel metadata should be output to the metadata file
use_kernel_code = true  # Convert kernel type string to kernel code (defined in root constants file)
sigma_x_range = [ 0.2, 3.0,]  # select a sigma-x value between 0.2-3.0
sigma_y_range = [ 0.2, 3.0,]  # select a sigma-y value between 0.2-3.0
normalize_metadata = true # normalize metadat values to the range 0-1
```

The individual parameters available for each type of operation are available in the docstrings of each ```__init__.py``` file for each degradation (under ```rumpy/image_tools```).

## Examples

Several example config files are provided in the ```sample_degradation_generators``` folder.  These files do not require any other CLI parameters to run e.g.:

```zsh
image_manipulate --pipeline_config blur_downsample_noise_compress.toml
```

## Outputs

Apart from each degraded image, ```image_manipulate``` will produce a copy of the provided degradation config file (```degradation_config.toml```), a list of degradation hyperparameters (```degradation_hyperparameters.csv```) and the specific degradations applied for each image (```degradation_metadata.csv```).  A ```degradation_metadata.csv``` file is required for training degradation predictors or blind SR models.
