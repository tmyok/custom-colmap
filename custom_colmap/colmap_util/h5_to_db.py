#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import h5py
import numpy as np
import os
import warnings

from PIL import Image, ExifTags
from tqdm import tqdm

from .database import image_ids_to_pair_id

def get_focal(image_path, err_on_default=False):
    image         = Image.open(image_path)
    max_size      = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35. * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal

def create_camera(db, image_path, camera_model, focal_length_prior=None):
    image         = Image.open(image_path)
    width, height = image.size

    if focal_length_prior is None:
        focal = get_focal(image_path)
    else:
        focal = focal_length_prior

    if camera_model == "simple-pinhole":
        model = 0 # simple pinhole
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == "pinhole":
        model = 1 # pinhole
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == "simple-radial":
        model = 2 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == "radial":
        model = 3 # simple radial
        param_arr = np.array([focal, width / 2, height / 2, 0.1, 0.1])
    elif camera_model == "opencv":
        model = 4 # opencv
        param_arr = np.array([focal, focal, width / 2, height / 2, 0., 0., 0., 0.])

    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, h5_path, image_path, camera_model, single_camera = False, focal_length_dict=None):
    keypoint_f = h5py.File(os.path.join(h5_path, "keypoints.h5"), "r")

    camera_id = None
    fname_to_id = {}
    for filename in tqdm(list(keypoint_f.keys()), total=len(keypoint_f.keys()), desc="Adding keypoints"):
        keypoints = keypoint_f[filename][()]

        fname_with_ext = filename
        path = os.path.join(image_path, fname_with_ext)
        if not os.path.isfile(path):
            raise IOError(f"Invalid image path {path}")

        if camera_id is None or not single_camera:
            if focal_length_dict is None:
                camera_id = create_camera(db, path, camera_model)
            else:
                camera_id = create_camera(db, path, camera_model, focal_length_dict[filename])
        image_id = db.add_image(fname_with_ext, camera_id)
        fname_to_id[filename] = image_id

        db.add_keypoints(image_id, keypoints)

    return fname_to_id

def add_matches(db, h5_path, fname_to_id):

    match_file = h5py.File(os.path.join(h5_path, "matches.h5"), "r")

    fmat_path = os.path.join(h5_path, "fMat.h5")
    if os.path.exists(fmat_path):
        fmat_file = h5py.File(fmat_path, "r")
    else:
        fmat_file = None

    added = set()
    for key_1 in tqdm(match_file.keys(), total=len(match_file.keys()), desc="Adding matches"):
        group = match_file[key_1]
        group2 = fmat_file[key_1] if fmat_file is not None else None

        for key_2 in group.keys():
            id_1 = fname_to_id[key_1]
            id_2 = fname_to_id[key_2]

            pair_id = image_ids_to_pair_id(id_1, id_2)
            if pair_id in added:
                warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!")
                continue

            matches = group[key_2][()]
            db.add_matches(id_1, id_2, matches)

            added.add(pair_id)

            if group2 is not None:
                fmat = group2[key_2][()]
                db.add_two_view_geometry(id_1, id_2, matches, fmat, np.eye(3), np.eye(3))

    return