import os
import ast
from pydoc import locate
from rumpy.shared_framework.configuration.constants import code_base_directory


ml_tasks = ['SISR', 'VSR', 'regression']  # all defined tasks within framework TODO: better place to define this?
available_models = {}

for task in ml_tasks:
    model_dir = os.path.join(code_base_directory, task, 'models')
    if not os.path.isdir(model_dir):
        continue
    # quick logic searching for all folders in models directory
    model_categories = [f.name for f in os.scandir(model_dir) if (f.is_dir() and '__' not in f.name)]
    # Main logic for searching for handler files and registering model architectures in system.
    for category in model_categories:
        handler_file = os.path.join(model_dir, category, 'handlers.py')
        if not os.path.isfile(handler_file):
            continue
        p = ast.parse(open(handler_file, 'r').read())
        classes = [node.name for node in ast.walk(p) if isinstance(node, ast.ClassDef)]
        for _class in classes:
            available_models[_class.split('Handler')[0].lower()] = ('rumpy.%s.models.' % task
                                                                    + category + '.handlers.' + _class)


def define_model(name, **kwargs):
    """
    Main model extractor.
    :param name: Model name.
    :param kwargs: Model params.
    :return: instantiated model architecture.
    """
    return locate(available_models[name])(**kwargs)
