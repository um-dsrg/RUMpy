import os
import ast
from rumpy.shared_framework.configuration.constants import code_base_directory


# quick logic searching for all folders in models directory  TODO: can this + the model searcher be converted into a fn?
image_tools_dir = os.path.join(code_base_directory, 'image_tools')
tool_categories = [f.name for f in os.scandir(image_tools_dir) if (f.is_dir() and '__' not in f.name)]
available_tools = {}

# Main logic for searching for degradation classes and registering them in the system
# all class names that contain 'Base' are ignored
for category in tool_categories:
    base_file = os.path.join(image_tools_dir, category, '__init__.py')
    if not os.path.isfile(base_file):
        continue
    p = ast.parse(open(base_file, 'r').read())
    classes = [node.name for node in ast.walk(p) if (isinstance(node, ast.ClassDef)) and ('Base' not in node.name)]
    for _class in classes:
        available_tools[_class.lower()] = ('rumpy.image_tools.' + category + '.' + _class)

