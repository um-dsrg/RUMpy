import face_alignment
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
from tqdm import tqdm


def align_dir(images_dir, file_name, save_loc, dim='2D', device='cpu', verbose=False, recursive=False):

    if dim == '2D':
        landmarks_type = face_alignment.LandmarksType._2D
    elif dim == '3D':
        landmarks_type = face_alignment.LandmarksType._3D
    else:
        raise Exception('Landmarks can either be 2D or 3D.')

    fa_net = face_alignment.FaceAlignment(landmarks_type, device=device, flip_input=True, verbose=verbose)
    landmarks = fa_net.get_landmarks_from_directory(images_dir, recursive=recursive)
    clean_landmarks = {}
    no_face_images = []
    for key, val in landmarks.items():
        new_key = key.split('/')[-1]
        if val is None:
            print('Face not found in:', new_key)
            data = 'Face not found'
            no_face_images.append(new_key)
        else:
            data = val[0]
        clean_landmarks[new_key] = data
    if len(no_face_images) > 0:
        with open(os.path.join(save_loc, 'no_face_images.csv'), "w") as f:
            writer = csv.writer(f)
            f.write('Images\n')
            for entry in no_face_images:
                writer.writerow([entry])
    pickle.dump(clean_landmarks, open(os.path.join(save_loc, file_name), "wb"))


def find_inter_eye_distances(landmarks):

    l_eye = slice(36, 42)
    r_eye = slice(42, 48)

    with open('../Data/celeba/inter_eye_dists_full.csv', 'w') as f:
        f.write('Index,Inter-Eye Distance\n')
        for index, (key, landmark) in enumerate(tqdm(landmarks.items())):
            if landmark is None:
                continue
            inter_eye_dist = np.linalg.norm(np.abs(np.mean(landmark[0][l_eye], 0) - np.mean(landmark[0][r_eye], 0)))
            f.write('{}, {}\n'.format(key.split('/')[-1].split('.')[0], inter_eye_dist))


def interpret_alignment():
    df = pd.read_csv('../Data/celeba/inter_eye_dists_full.csv')

    df.hist(column='Inter-Eye Distance', bins=200)
    plt.title('Celeba Inter-Eye Distances')
    plt.show()

    df.plot(kind='scatter', x='Index', y='Inter-Eye Distance')
    plt.title('Celeba Inter-Eye Distances vs Image Index')
    plt.show()
    return df

