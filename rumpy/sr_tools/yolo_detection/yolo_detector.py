import os
import glob
import csv
import sys
from tqdm import tqdm
from multiprocessing import Process
import click
import rumpy.shared_framework.configuration.constants as sconst
from rumpy.sr_tools.yolo_detection.yolo_utils import *


# Based on https://github.com/sthanhng/yoloface
class YoloDetector:
    def __init__(self):
        model_cfg = './sr_tools/yolo_detection/yolov3-face.cfg'
        model_weights = './sr_tools/yolo_detection/yolov3-wider_16000.weights'
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def run_detection(self, image, mark_boundary=False):
        cap = cv2.VideoCapture(image)
        has_frame, frame = cap.read()
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(get_outputs_names(self.net))

        if mark_boundary:
            faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

        return frame.astype(np.uint8), self.extract_face(frame, outs, CONF_THRESHOLD)

    def extract_face(self, frame, outs, conf_threshold):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
        return boxes


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def process_images(images, output_name):
    detector = YoloDetector()
    with open(output_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'left', 'top', 'width', 'height'])
        for image in tqdm(images):
            _, boundary = detector.run_detection(image)
            if len(boundary) == 0:
                writer.writerow([image.split('/')[-1], 'Not Detected'])
            else:
                writer.writerow([image.split('/')[-1]] + boundary[0])


@click.command()
@click.option("--input_dir", default=os.path.join(sconst.data_directory, 'celeba/png_samples/celeba_png_align/eval_samples'),
              help='Input directory to source images.')
@click.option("--splits", default=1,
              help='Number of processes to spawn for multiprocessing speedup.')
def process_folder(input_dir, splits):
    images = []
    for extension in ['*.jpg', '*.png', '*.bmp']:
        images.extend(glob.glob(os.path.join(input_dir, extension)))
    images.sort()
    groups = chunks(images, int(len(images)/splits))
    procs = []
    # Multiprocessing code pointless, as package already takes care of multiprocessing by itself!
    # Can use similar structure in other places
    for i, group in enumerate(groups):
        print("Starting group {}".format(i))
        proc = Process(target=process_images, args=(group, os.path.join(input_dir, 'face_boundaries_%d.csv' % i)))
        procs.append(proc)
        proc.start()


if __name__ == '__main__':
    process_folder(sys.argv[1:])
