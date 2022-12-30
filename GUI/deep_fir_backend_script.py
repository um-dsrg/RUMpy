import sys
import click
import time
import numpy as np
import PIL.Image
import onnxruntime as ort

@click.command()
@click.option("--model",
              help='ONNX model to run the inference.')
@click.option("--source_image",
              help='Location of image to run.')
@click.option("--save_image",
              help='Location of image to save.')
@click.option("--use_gpu", is_flag=True,
              help='Add flag to use GPU instead of the default CPU.')

def run_onnx_model(**kwargs):
    if 'use_gpu' in kwargs and kwargs['use_gpu']:
        provider = ['CUDAExecutionProvider']
    else:
        provider = ['CPUExecutionProvider']

    start = time.time()

    print('Setting optimization options...')
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    print('Creating inference session...')
    ort_sess = ort.InferenceSession(kwargs['model'],
                                    providers=provider,
                                    sess_options=sess_options)

    print('Loading image...')
    PIL_image = PIL.Image.open(kwargs['source_image'])
    np_image = np.asarray(PIL_image)
    np_image_reshape = np.expand_dims(np.rollaxis(np_image, 2).astype(np.float32), 0) / 255.0

    print('Running inference...')
    output = ort_sess.run(None, {'input': np_image_reshape})[0]
    np_output_squeeze = np.moveaxis(np.squeeze(output), 0, 2)
    PIL_output = PIL.Image.fromarray(np.clip(np_output_squeeze*255, 0, 255).astype(np.uint8))

    print('Saving image...')
    PIL_output.save(kwargs['save_image'])

    end = time.time()
    print('Total time:', end-start)

if __name__ == '__main__':
    run_onnx_model(sys.argv[1:])

    # To turn this script into an executable:
    # pyinstaller deep_fir_backend_script.py -F -n DeepFIR --clean