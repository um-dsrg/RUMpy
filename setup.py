""" Setup script for package. """
from setuptools import setup, find_packages

setup(
    name="RUMpy",
    author="Matthew Aquilina, Keith George Ciantar, Christian Galea",
    version='1.0',
    url="https://github.com/um-dsrg/RUMpy",
    description="Machine learning package containing functionality for creating, "
                "training and validating a variety of SISR, regression and VSR models.",
    packages=find_packages(),
    include_package_data=True,
    license='GPL-3.0',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Machine Learning',

        # Pick your license as you wish (should match "license" above)
        'License :: GPL-3.0 License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points='''
        [console_scripts]
        train_sisr=rumpy.shared_framework.net_train:experiment_setup
        eval_sisr=rumpy.shared_framework.net_eval:eval_run
        image_manipulate=rumpy.image_tools.click_cli:image_manipulator
        find_faces=rumpy.sr_tools.yolo_detection.yolo_detector:process_folder
        face_segment = rumpy.sr_tools.face_segmentation.segmentation:segment
        images_to_video = rumpy.sr_tools.helper_functions:click_image_sequence_to_movie
        extract_best_model = rumpy.sr_tools.helper_functions:extract_best_models
        clean_models = rumpy.sr_tools.helper_functions:click_clean
        model_report = rumpy.sr_tools.helper_functions:model_compare
    ''',
)

