# TODO: This contains old functions contained within the Basemodel that used to be implemented to allow models to focus on faces only when training.
# However, this is currently deprecated, and should only be inserted back into the general models when updated.


def __init__(self, device, model_save_dir, eval_mode, grad_clip=None, face_training=False,
             hr_data_loc=None, mask_type='bisenet_background', **kwargs):
    """
    :param device: GPU device ID (or 'cpu').
    :param model_save_dir: Model save directory.
    :param eval_mode: Set to true to turn off training functionality.
    :param grad_clip: If gradient clipping is required during training, set gradient limit here.
    :param face_training:  Set to true if training with face segmentation masks is required.
    :param hr_data_loc: HR image data location.
    :param mask_type: Face segmentation mask type (bisenet/bisenet_background/yolo).
    """
    super(BaseModel, self).__init__()
    self.criterion = nn.L1Loss()
    if device == 'cpu':
        self.device = torch.device('cpu')
    else:
        self.device = device
    self.optimizer = None  # - |
    self.net = None  # | defined in specific architectures
    self.face_finder = False  # |
    self.model_name = None  # - |
    self.im_input = None  # - |
    self.colorspace = None  # - |
    if grad_clip == 0:
        self.grad_clip = None
    else:
        self.grad_clip = grad_clip
    self.model_save_dir = model_save_dir
    self.eval_mode = eval_mode
    self.curr_epoch = 0
    self.state = {}
    self.learning_rate_scheduler = None
    self.legacy_load = True  # loading system which ensures weight names match as expected

    if face_training:
        self.mask_mode = mask_type
        self.face_finder = True
        if mask_type == 'yolo':
            self.boundary_data = pd.read_csv(os.path.join(hr_data_loc, 'face_boundaries_0.csv'),
                                             header=0, index_col=0, squeeze=True).dropna().astype(int).to_dict(
                'index')
            marked = []
            for k, v in self.boundary_data.items():  # wipes off any entries with negative numbers TODO: make a function for this
                if any(v_in < 0 for v_in in v.values()):
                    marked.append(k)
            for k in marked:
                self.boundary_data.pop(k, None)

    self.accepted_masks = torch.tensor([[0, 85, 255],
                                        [0, 255, 0],
                                        [170, 0, 255],
                                        [255, 0, 0],
                                        [0, 85, 255],
                                        [255, 0, 170],
                                        [255, 0, 85],
                                        [255, 85, 0]])
    """
    [0,85,255] = face - accept
    [0, 255, 0] = left eye - accept
    [170, 255, 255] = hair/clothes
    [170, 0, 255] = right eye - accept
    [255, 0, 0] = nose - accept
    [0, 85, 255] = right eyebrow - accept
    [170, 255, 255] = left eyebrow - accept
    [0, 255, 170] = right ear?
    [255, 170, 0] = neck
    [255, 0, 170] = top lip - accept
    [255, 0, 85] = mouth? - accept
    [255, 85, 0] = lower lip - accept
    [170, 255, 0] = bottom ear?
    [255, 0, 255] = hat
    """


def run_train(self, x, y, tag=None, mask=None, keep_on_device=False, *args, **kwargs):
    if self.eval_mode:
        raise RuntimeError('Model initialized in eval mode, training not possible.')
    self.net.train()  # sets model to training mode (activates appropriate procedures for certain layers)
    x, y = x.to(device=self.device), y.to(device=self.device)
    out = self.run_model(x, image_names=tag, **kwargs)  # run data through model

    if self.face_finder:
        face_mask = self.get_face_mask(y, image_names=tag, masks=mask).to(device=self.device)
        y = y * face_mask
        out = out * face_mask

    loss = self.criterion(out, y)  # compute loss
    self.standard_update(loss)  # takes care of optimizer calls, backprop and scheduler calls

    if keep_on_device:
        return loss.detach().cpu().numpy(), out.detach()
    else:
        return loss.detach().cpu().numpy(), out.detach().cpu()

def get_face_mask(self, y, image_names=None, masks=None):
    face_crop = torch.zeros(y.size(0), 3, *y.size()[2:])

    if self.mask_mode == 'yolo':
        for index, image in enumerate(image_names):
            if image in self.boundary_data:
                box = self.boundary_data[image]
                face_crop[index, :, box['top']:box['top'] + box['height'],
                box['left']:box['left'] + box['width']] = 1
            else:
                face_crop[index, ...] = 1
    elif self.mask_mode == 'bisenet':
        for index, mask in enumerate(masks):  # TODO: this could be fully optimized - removing the for loop
            face_crop[index, ...] = (mask == self.accepted_masks[:, None, None, :]).all(3).any(0)
    elif self.mask_mode == 'bisenet_background':
        for index, mask in enumerate(masks):
            face_crop[index, ...] = ~(mask == torch.tensor((255, 255, 255))).all(2)
    return face_crop
