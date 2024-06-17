import cv2
import h5py
import kornia as K
import torch

from lightglue import ALIKED
from tqdm.auto import tqdm

def resize(image, image_size):
    h, w = image.shape[:2]
    aspect_ratio = h/w
    smaller_side_size = int(image_size/max(aspect_ratio, 1/aspect_ratio))
    if aspect_ratio > 1: # H > W
        new_size = (image_size, smaller_side_size)
    else: # H <= W
        new_size = (smaller_side_size, image_size)
    image = cv2.resize(image, new_size[::-1], interpolation=cv2.INTER_AREA)
    return image, new_size

def detect_keypoints_ALIKED(
    image_paths,
    feature_dir,
    device,
    resize_to,
):
    dtype = torch.float32 # ALIKED has issues with float16

    extractor = ALIKED().eval().to(device, dtype)

    with h5py.File(feature_dir / "keypoints.h5", mode="w") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", mode="w") as f_descriptors:

        for path in tqdm(image_paths, desc=f"Computing keypoints", dynamic_ncols=True):

            image = cv2.imread(str(path))
            original_size = image.shape[:2]
            image, new_size = resize(image, resize_to)
            image = K.image_to_tensor(image, False).float() / 255.
            image = K.color.bgr_to_rgb(image).to(device).to(dtype)

            with torch.inference_mode():
                features = extractor.extract(image, resize=None)

            keypoints = features["keypoints"].squeeze().detach().cpu().numpy()
            keypoints = keypoints * original_size / new_size

            key = path.name
            f_keypoints[key] = keypoints
            f_descriptors[key] = features["descriptors"].squeeze().detach().cpu().numpy()

    return

if __name__ == "__main__":

    import argparse
    import matplotlib.pyplot as plt
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, default="../output/debug")
    parser.add_argument("--resize_to", type=int, default=2048)
    parser.add_argument("--draw_results", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_dir = args.output_dir / ".features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    detect_keypoints_ALIKED(
        image_paths=list(args.image_dir.glob("*")),
        feature_dir=feature_dir,
        device=device,
        resize_to=args.resize_to,
    )

    # draw keypoints
    if args.draw_results:

        with h5py.File(feature_dir / "keypoints.h5", mode='r') as f_keypoints:

            for key in tqdm(f_keypoints.keys(), desc="Drawing keypoints", dynamic_ncols=True):

                keypoints = f_keypoints[key][...]

                image = cv2.imread(str(args.image_dir / key))

                plt.figure(figsize=(60, 40))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.scatter(keypoints.squeeze()[:, 0], keypoints.squeeze()[:, 1], s=10, c='r', marker='o')

                title = key
                title += f'\n# of keypoints: {keypoints.shape[0]}'
                plt.title(title)
                plt.tight_layout()
                plt.savefig(args.output_dir / key)
                plt.clf()
                plt.close()