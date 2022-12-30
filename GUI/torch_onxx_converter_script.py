import os
import sys
import click
import torch
import onnx
import onnxoptimizer
import onnxsim
from rumpy.SISR.models.interface import SISRInterface

@click.command()
@click.option("--input_model_location",
              help='Parent folder where the PyTorch model folder is found.')
@click.option("--input_model_name",
              help='PyTorch model save folder.')
@click.option("--output_model_location",
              help='Folder to save the converted model.')
@click.option("--output_model_name",
              help='Name of the converted model (if .onnx is not included it will be added automatically).')
@click.option("--load_epoch", default='best',
              help='Which epoch to load, normally either \'best\' or \'last\'.')
@click.option("--scale", default=4,
              help='Model upsampling factor.')
@click.option("--optimize", is_flag=True,
              help='Run the ONNX optimizer on the model.')
@click.option("--simplify", is_flag=True,
              help='Run the ONNX simplifier on the model.')

def run_model_converter(**kwargs):
    # Load the SISR model interface
    model_interface = SISRInterface(kwargs['input_model_location'],
                                    kwargs['input_model_name'],
                                    load_epoch=kwargs['load_epoch'],
                                    gpu='off',
                                    scale=kwargs['scale'])

    # Get the network and set it to eval mode
    model = model_interface.model.net
    model.eval()

    # Input to the model
    sample_input = torch.randn(1, 3, 100, 100, requires_grad=True)

    # Set the dynamic axes to allow models of varying height and width
    # NOTE: can't set more than 3 dynamic axes, seems like an ONNX limitation
    dynamic_axes = {'input': [2, 3],
                    'output': [2, 3]}

    model_save_name = kwargs['output_model_name']
    if not model_save_name.endswith('.onnx'):
        model_save_name = model_save_name + '.onnx'

    # Export the model
    torch.onnx.export(model,
                      sample_input,              # model input (can be a tuple for multiple inputs)
                      os.path.join(kwargs['output_model_location'], model_save_name),
                      export_params=True,
                      opset_version=15,          # the ONNX version to export the model to (higher means more recent)
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes=dynamic_axes)

    find_extension = model_save_name.find('.onnx')
    onnx_model = onnx.load(os.path.join(kwargs['output_model_location'], model_save_name))

    if kwargs['optimize']:
        optimized_model_save_name = model_save_name[:find_extension] + '_optimized' + model_save_name[find_extension:]
        optimized_model = onnxoptimizer.optimize(onnx_model, ['fuse_bn_into_conv',
                                                              'fuse_add_bias_into_conv',
                                                              'fuse_pad_into_conv',
                                                              'fuse_consecutive_squeezes',
                                                              'fuse_consecutive_concats',
                                                              'fuse_consecutive_reduce_unsqueeze',
                                                              'eliminate_deadend'])
        onnx.save(optimized_model, os.path.join(kwargs['output_model_location'], optimized_model_save_name))

    if kwargs['simplify']:
        simplified_model, check = onnxsim.simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        simplified_model_save_name = model_save_name[:find_extension] + '_simplified' + model_save_name[find_extension:]
        onnx.save(simplified_model, os.path.join(kwargs['output_model_location'], simplified_model_save_name))

if __name__ == '__main__':
    run_model_converter(sys.argv[1:])

    # Example script:
    # python torch_onxx_converter_script.py --input_model_location <div2k> --input_model_name rcan_cosine_v2 --output_model_location <cwd> --output_model_name <model.onnx> --load_epoch best --optimize --simplify