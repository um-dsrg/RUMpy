import base64
import io
from io import BytesIO

import numpy as np
import torch
import torchvision.transforms.functional as F
from flask import Flask, request
from PIL import Image
from torchvision import transforms

from rumpy.SISR.models.interface import SISRInterface


class ServerHub:
    def __init__(self):
        self.model = None
        self.data_transform = transforms.ToTensor()
        self.scale = 4

    def pre_upsample(self, pil_img):
        pil_img = pil_img.resize((pil_img.width * self.scale, pil_img.height * self.scale), resample=Image.BICUBIC)

        return pil_img

    def load_image_b64(self, b64_string):
        base64_decoded = base64.b64decode(b64_string)
        img_1 = Image.open(io.BytesIO(base64_decoded))
        img = np.array(img_1)
        pil_img = Image.fromarray(np.uint8(img))

        if pil_img.mode == 'RGBA' or pil_img.mode == 'L':
            pil_img = pil_img.convert('RGB')

        # TODO: Add a way to choose whether to pre-upsample or not
        # pil_img = self.pre_upsample(pil_img)
        tensor_img = self.data_transform(pil_img)

        return tensor_img, pil_img

    def crop_image_tensor(self, tensor_image, x, y, width, height):
        return F.crop(tensor_image, round(y), round(x), round(height), round(width))

    def crop_image_PIL(self, PIL_image, x, y, width, height):
        return PIL_image.crop((round(x), round(y), round(x)+round(width), round(y)+round(height)))

    def get_metadata_from_string(self, metadata_string):
        metadata_array = np.fromstring(metadata_string, dtype=float, sep=',')
        metadata_array_reshape = np.reshape(metadata_array, (1, len(metadata_array)))

        tensor_metadata = torch.from_numpy(metadata_array_reshape)

        return tensor_metadata, metadata_array

    def normalise_qpi(self, qpi_val, min_val=20, max_val=40):
        return (qpi_val - min_val) / (max_val - min_val)

    def get_metadata_from_two_strings(self, blur_kernel_string, qpi_string):
        metadata_array = np.append(np.fromstring(blur_kernel_string, dtype=float, sep=','), self.normalise_qpi(float(qpi_string)))
        metadata_array_reshape = np.reshape(metadata_array, (1, len(metadata_array)))

        tensor_metadata = torch.from_numpy(metadata_array_reshape)

        return tensor_metadata, metadata_array

    def convert_image_b64(self, np_img):
        buff = BytesIO()
        uint8_image = (np_img.transpose([1, 2, 0])*255).astype(np.uint8)
        pil_img = Image.fromarray(uint8_image)
        pil_img.save(buff, format='PNG')
        b64_img = base64.b64encode(buff.getvalue()).decode('ascii')
        return b64_img

    def convert_bicubic_image_b64(self, np_img):
        buff = BytesIO()
        uint8_image = np.uint8(np_img)
        pil_img = Image.fromarray(uint8_image)
        pil_img.save(buff, format='PNG')
        b64_img = base64.b64encode(buff.getvalue()).decode('ascii')
        return b64_img

app = Flask(__name__)

@app.route('/super_resolve', methods=['POST'])
def super_resolve():
    image_b64 = request.form['image']
    tensor_image, _ = server_hub.load_image_b64(image_b64)

    metadata_keys = None
    tensor_metadata = None
    result = None

    # TODO: Maybe have a way where the model is loaded depending on the data
    if 'blur_kernel' in request.form:
        if 'QPI' in request.form:
            blur_kernel_string = request.form['blur_kernel']
            qpi_string = request.form['QPI']

            tensor_metadata, metadata_array = server_hub.get_metadata_from_two_strings(blur_kernel_string, qpi_string)

            metadata_keys = [('blur_kernel',) for m in metadata_array]
            metadata_keys[-1] = ('qpi',)
            result, *_ = server_hub.model.net_run_and_process(lr=tensor_image.unsqueeze(0), hr=None, metadata=tensor_metadata, metadata_keys=metadata_keys)
        else:
            blur_kernel_string = request.form['blur_kernel']
            tensor_metadata, metadata_array = server_hub.get_metadata_from_string(blur_kernel_string)

            metadata_keys = [('blur_kernel',) for m in metadata_array]
            result, *_ = server_hub.model.net_run_and_process(lr=tensor_image.unsqueeze(0), hr=None, metadata=tensor_metadata, metadata_keys=metadata_keys)
    else:
        result, *_ = server_hub.model.net_run_and_process(lr=tensor_image.unsqueeze(0))

    b64_sr_image = server_hub.convert_image_b64(result.squeeze())

    return b64_sr_image

@app.route('/super_resolve_and_crop', methods=['POST'])
def super_resolve_with_crop():
    image_b64 = request.form['image']
    tensor_image, _ = server_hub.load_image_b64(image_b64)

    if 'x' in request.form:
        tensor_image = server_hub.crop_image_tensor(tensor_image,
                                                    float(request.form['x']),
                                                    float(request.form['y']),
                                                    float(request.form['width']),
                                                    float(request.form['height']))

    metadata_keys = None
    tensor_metadata = None
    result = None

    # TODO: Maybe have a way where the model is loaded depending on the data
    if 'blur_kernel' in request.form:
        if 'QPI' in request.form:
            blur_kernel_string = request.form['blur_kernel']
            qpi_string = request.form['QPI']

            tensor_metadata, metadata_array = server_hub.get_metadata_from_two_strings(blur_kernel_string, qpi_string)

            metadata_keys = [('blur_kernel',) for m in metadata_array]
            metadata_keys[-1] = ('qpi',)
            result, *_ = server_hub.model.net_run_and_process(lr=tensor_image.unsqueeze(0), hr=None, metadata=tensor_metadata, metadata_keys=metadata_keys)
        else:
            blur_kernel_string = request.form['blur_kernel']
            tensor_metadata, metadata_array = server_hub.get_metadata_from_string(blur_kernel_string)

            metadata_keys = [('blur_kernel',) for m in metadata_array]
            result, *_ = server_hub.model.net_run_and_process(lr=tensor_image.unsqueeze(0), hr=None, metadata=tensor_metadata, metadata_keys=metadata_keys)
    else:
        result, *_ = server_hub.model.net_run_and_process(lr=tensor_image.unsqueeze(0))

    b64_sr_image = server_hub.convert_image_b64(result.squeeze())

    return b64_sr_image

@app.route('/super_resolve_bicubic', methods=['POST'])
def super_resolve_bicubic():
    image_b64 = request.form['image']
    _, pil_image = server_hub.load_image_b64(image_b64)

    bicubic_image = server_hub.pre_upsample(pil_image)
    b64_bicubic_image = server_hub.convert_bicubic_image_b64(np.array(bicubic_image))

    return b64_bicubic_image

@app.route('/super_resolve_bicubic_and_crop', methods=['POST'])
def super_resolve_bicubic_with_crop():
    image_b64 = request.form['image']
    _, pil_image = server_hub.load_image_b64(image_b64)

    if 'x' in request.form:
        pil_image = server_hub.crop_image_PIL(pil_image,
                                              float(request.form['x']),
                                              float(request.form['y']),
                                              float(request.form['width']),
                                              float(request.form['height']))

    bicubic_image = server_hub.pre_upsample(pil_image)
    b64_bicubic_image = server_hub.convert_bicubic_image_b64(np.array(bicubic_image))

    return b64_bicubic_image

@app.route('/update_model', methods=['POST'])
def update_model():
    server_hub.model = SISRInterface(request.form['location'],
                                     request.form['name'],
                                     load_epoch=request.form['epoch'] if not request.form['epoch'].isdigit() else int(request.form['epoch']),
                                     gpu='single' if request.form['processor'].lower() == 'gpu' else 'off',
                                     scale=4)

    return request.form['name']

@app.route('/test_page')
def test():
    return 'OK'

if __name__ == '__main__':
    server_hub = ServerHub()
    app.run(host='127.0.0.1', port=5000)
