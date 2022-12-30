Deep-FIR GUI
==========================

The GUI application was designed to have a more interactive way of using the codebase, and playing around with trained super-resolution models. The current system makes use of a web-based frontend to load and manipulate images/videos, and a server-based backend to carry out super-resolution. This configuration allows for separation of responsibility and also gives flexibility in how the system can be configured.

Frontend User Interface - Electron
--------------------------

The current frontend is built with [Electron](https://www.electronjs.org/), which allows us to create a desktop application using web technologies. With this framework, the frontend has a lot of flexibility, as it can be built using HTML, CSS and JavaScript. Rendering is done using an embedded [Chromium](https://www.chromium.org/chromium-projects/) browser engine, while the [Node.js](https://nodejs.org/en/) runtime environment is used as the backend.

### Installation

Before installing the application itself, make sure that the **Node Package Manager (npm)** is installed by downloading a recent version of Node.js from [the official website](https://nodejs.org/en/download/). Once installed, ensure that everything is working correctly by opening a terminal and entering the commands ```npm -v``` and ```node -v```.

After this, navigate to the ```./GUI``` folder, open a terminal and run the ```npm install``` command. This will make use of the ```package.json``` and ```package-lock.json``` to install the required packages which, will be stored in the ```./node_modules``` folder.

As a quick check to see that everything has been installed correctly, run the ```npm start``` command and navigate through the application. The home page should show a high-level overview of the project, and there should be a navigation bar on the left, to access the other pages.

### Packaging the Electron Application

To build the application as a binary for different platforms, the [electron-packager](https://github.com/electron/electron-packager) tool is used. This packager can either be used a command-line tool or a Node.js library, and greatly simplifies the process of bundling an Electron executable and supporting files into a folder that is ready for distribution. In the ```package.json``` file, this package is used in the build command ```rimraf DeepFIR_GUI-* && electron-packager . --platform=darwin,win32,linux --arch=x64 --icon=app```, which is set up to generate files for Windows, Mac and Linux, and set the thumbnails using their corresponding app icons.

To run the application packager, open a terminal in the ```./GUI``` folder and run the command ```npm run build```. This should take a few minutes and will result in three folders in the form of ```./DeepFiR_GUI-*OS name*-x64```. The respective executables can be found in each folder and the application can be run by either clicking on the file, or in the case of Linux running the command ```./DeepFIR_GUI-linux-x64/DeepFIR_GUI```.

Backend Super-Resolution System - Flask
--------------------------

The current backend system uses the [Flask](https://flask.palletsprojects.com/en/2.2.x/) framework to run a server that hosts the super-resolution API. The application is written in Python, and makes use of the RUMpy package (along with some others) to take requests from the frontend and run the SR models on low-resolution images.

### Installation

To install the packages for this system, first follow the installation steps outlined in the main [README](../README.md/#installation) file. This should ensure that the packages used by the codebase are contained in a conda environment, and that the RUMpy package has been installed. Additionally, in the same environment also install the Flask package with the command ```conda install flask```, which will be used for our backend script.

### Packaging the Backend Script

At the moment there is no straightforward way to package this script. After numerous attempts with various packaging methods, it seems that the combination of packages and the preferred package installer make it rather difficult to generate an executable for the server backend. However, due to time constraints, we did not manage to thoroughly try packaging with newer versions of Python or with other package installers (e.g., pip), so a solution could still be feasible. Until then, we have opted to run the system as a script in development mode.

Execution
--------------------------

To operate the application with both the Electron frontend and the Super-Resolution API backend, the following steps need to be carried out.

1. Open the ```./GUI``` folder, and check for the file named ```models.csv```. If this file is not found, create it yourself. The file should contain a table-like list, with information on the models that are downloaded on your system. The file ```models_template.csv``` is given as an example of how ```models.csv``` should be populated, as shown below.
    ```
    name,label,location,epoch,group
    model_1,"Model One","path/to/model/folder",best,"Type 1"
    model_2,"Model Two","path/to/model/folder",best,"Type 2"
    ```
2. Open a terminal in the ```./GUI``` folder and run the ```npm start``` command. *Optionally, after packaging with the ```npm run build``` command, click on the resulting executable in one of the folders with the name ```DeepFIR_GUI-*OS name*-x64```.*
3. Open another terminal in the same folder and activate the DeepFIR virtual environment (typically using a command like ```conda activate *ENV_NAME*```, where ```*ENV_NAME*``` is the name of the environment).
4. Run the command ```python deep_fir_server.py``` or ```python3 deep_fir_server.py``` and wait until the server has finished setting up.
5. Go through the Electron application and check that there are no issues with the pages. To carry out super-resolution on an image or a video frame, make sure to first read through the instructions, to get an overview of the necessary steps.

Instruction Video
--------------------------

[![Watch the video](https://img.youtube.com/vi/sptjtztnQnE/maxresdefault.jpg)](https://youtu.be/sptjtztnQnE)

Ideas for further development
==========================

New Proposed Backend - ONNX (Currently unused)
--------------------------

[ONNX](https://github.com/onnx/onnx) is an open ecosystem that supports a wide range of frameworks (PyTorch, TensorFlow, Keras, etc.) and provides an open source format for deep learning and machine learning models. This system is natively supported in PyTorch via ```torch.onnx```, and greatly simplifies the process of exporting the models into a portable format, and running them without needing PyTorch.

### Converting trained PyTorch models to ONNX

#### Installation

Before running any scripts, make sure to run ```pip install onnxoptimizer onnxsim``` so that both the ONNX optimizer and simplifier are installed. These packages are used to try to squeeze out some performance improvements from the converted ONNX model.

To convert a trained PyTorch model into ONNX format, the command-line script ```torch_onxx_converter_script.py``` can be used. If everything is working correctly, running the script with the ```--help``` command should give the following output.

```
Usage: torch_onxx_converter_script.py [OPTIONS]

Options:
  --input_model_location TEXT   Parent folder where the PyTorch model folder
                                is found.
  --input_model_name TEXT       PyTorch model save folder.
  --output_model_location TEXT  Folder to save the converted model.
  --output_model_name TEXT      Name of the converted model (if .onnx is not
                                included it will be added automatically).
  --load_epoch TEXT             Which epoch to load, normally either 'best' or
                                'last'.
  --scale INTEGER               Model upsampling factor.
  --optimize                    Run the ONNX optimizer on the model.
  --simplify                    Run the ONNX simplifier on the model.
  --help                        Show this message and exit.
```

#### Example

To give an example of how to convert a PyTorch model into ONNX, one of the trained RCAN models will be used. To convert the ```rcan_cosine_v2``` model and also run optimizer and simplifier, the following command can be run.

```
python torch_onxx_converter_script.py --input_model_location /path/to/parent/folder --input_model_name rcan_cosine_v2 --output_model_location /path/to/output/folder --output_model_name rcan_cosine_v2.onnx --load_epoch best --scale 4 --optimize --simplify
```

This will generate three separte files: ```rcan_cosine_v2.onnx```,  ```rcan_cosine_v2_optimized.onnx``` and ```rcan_cosine_v2_simplified.onnx```, all stored in the specified location.

At the moment, the optimized version seems to have the best performance improvement out of the three, but there might room for more gains, if further tweaks are carried out.

#### Current optimizations and possible additions

At the time or writing, the ONNX models get optimized with the following code.

```
optimized_model = onnxoptimizer.optimize(onnx_model,
                                         ['fuse_bn_into_conv',
                                          'fuse_add_bias_into_conv',
                                          'fuse_pad_into_conv',
                                          'fuse_consecutive_squeezes',
                                          'fuse_consecutive_concats',
                                          'fuse_consecutive_reduce_unsqueeze',
                                          'eliminate_deadend'])
```
These options try to reduce and fuse certain operations, to speed-up the model inference. A variety of combinations have been tested, but so far, these are the ones that seem to be compatible with the test model.

For additional optimizations, the different options can be found in the [```passes``` sub-directory of the ONNX optimizer GitHub repository](https://github.com/onnx/optimizer/tree/master/onnxoptimizer/passes). Until now, there doesn't seem to be an official documentation source, so it's a matter or trial and error for which options/passes to use.

### Running the ONNX model

#### Installation

Before running any scripts with the converted ONNX models make sure to run ```pip install onnxruntime-gpu```. This will install two versions of the ONNX runtime, one for the CPU and one for the GPU. At the moment, testing was mostly done with the former, as the latter seems to require additional tweaks with GPU libraries.

To run the ONNX model, the command-line script ```deep_fir_backend_script.py``` can be used. This basic script takes the locations of the model, input image and save location, and runs the SISR system using the ONNX runtime.Once everything has been installed correctly, running with the ```--help``` option will give the following output.

```
Usage: deep_fir_backend_script.py [OPTIONS]

Options:
  --model TEXT         ONNX model to run the inference.
  --source_image TEXT  Location of image to run.
  --save_image TEXT    Location of image to save.
  --use_gpu            Add flag to use GPU instead of the default CPU.
  --help               Show this message and exit.
```

#### Example

To give a demo of this script, the same model mentioned during the format conversion (```rcan_cosine_v2.onnx```) will be used. For simplicity, the image names will be set arbitrarily.

```
python deep_fir_backend_script.py --model /path/to/output/folder/rcan_cosine_v2.onnx --source_image /path/to/div2k/LR/0846.png --save_image out.png
```

This should take a few seconds to run on the CPU and will generate a high quality image that has been upsampled by a factor of 4. Unfortunately, the ```--use_gpu``` flag doesn't seem to work at the moment, but might give some improvements with some tweaking of the CUDA version/paths.

### Packaging the new system

The benefit of using this system is that the script is kept very lightweight, as the Deep-FIR codebase and its dependencies don't need to be packaged. To create the executable, the [PyInstaller](https://pyinstaller.org/en/stable/) package is used, as it seems to be the most versatile in terms of cross-platform compatibility and popularity. This package can be installed using the command ```pip install pyinstaller```.

After all the packages have been installed, the following command can be run.
```
pyinstaller deep_fir_backend_script.py -F -n DeepFIR_inference --clean
```
This will create the ```DeepFIR_inference.exe``` in the ```dist``` folder, and can be used with the same arguments of the Python script. For now, the name of the ```.exe``` has been chosen arbitrarily, and could be changed to anything.

Important notes, unfinished features and new ideas
--------------------------

Here are some aspects and features from the system that are either still in development or that can be included in future versions.

- Updates to pages/menus/layout
  - Adding a hamburger icon/button to toggle the sidebar menu.
  - Updating the ```Settings``` page to allow the user to download new models (maybe from something like [Zenodo](https://zenodo.org/)).
  - The current layout (which goes through the SR process by scrolling through the page) is still a little experimental and might be a bit outdated. It could be updated by tweaking the HTML and CSS, and hiding certain panes when they're not in use.
- Updates to the SR system
  - The interpolation system used for the HTML canvas might be messing with the clarity of the results. Further testing and tweaking might be needed, to see how best to display the super-resolved and bicubically upsampled images, for a side-by-side comparison.
  - As a quality of life update, it would be useful to add an option to run a batch of images. This might need to be added as an option in the ```Settings``` or something of the sort, but it would be a good way to speed-up the user's experience instead of running one image at a time.
  - Improving the current method of saving results and showing comparisons. At the moment, a JavaScript package is used to save the results to a PDF. The first change which could be done, is to adjust the size and positioning of the plots, to make them easier to compare. It would also be helpful to add elements such as: image file names, SR model used, and general details to help keep track.
- Updates to the backend
  - The new ONNX system should be able to replace the Flask server, as it is much easier to package and run. To run a shell script through JavaScript, the following code from [StackOverflow](https://stackoverflow.com/questions/3152482/running-exe-from-javascript/3152512#3152512) could be used.
  - With the new ONNX runtime script, the Flask system could still be used, although it would be much easier to just run the executable directly.
  - **When converting a Pytorch model into an ONNX model, the export script allows the user to define ```dynamic_axes```. At the moment these have been set to [2, 3], which refer to the width and height of the image. When testing the conversion, many combinations with three axes (e.g., [0, 2, 3]) were tried, but it seems they're not supported by ONNX. Running multiple same-sized images in a batch (e.g., cropping a large image into patches) might help improve performance, but until there is some sort of update, we would need to decide on whether to choose a dynamic batch size, or dynamic image dimensions.**
- Features which might require more time and efforts
  - Updated frame selection and cropping system. This seems like a pretty complex task, but ideally the user could choose multiple frames from a video and send them to the SR backend.
  - Adding a way to have recently-saved images appear in a panel or carousel. This might need some re-working of the layout but it would be an interesting feature to have.