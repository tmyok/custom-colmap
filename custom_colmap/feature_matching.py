import cv2
import h5py
import kornia.feature as KF
import torch

from tqdm.auto import tqdm

from .get_image_pairs import get_image_pairs

def feature_matching_aliked_lightglue(
    image_paths,
    index_pairs,
    feature_dir,
    device,
):
    matcher_params = {
        "filter_threshold": 0.2,
        "width_confidence": -1,
        "depth_confidence": -1,
        "mp": True,
    }
    matcher = KF.LightGlueMatcher(
        feature_name="aliked",
        params=matcher_params,
    ).to(device).eval()

    with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, \
         h5py.File(feature_dir / "descriptors.h5", mode="r") as f_descriptors, \
         h5py.File(feature_dir / "matches.h5", mode="w") as f_matches:

            for idx1, idx2 in tqdm(index_pairs, desc="Feature matching", dynamic_ncols=True):
                key1, key2 = image_paths[idx1].name, image_paths[idx2].name

                keypoints1 = torch.from_numpy(f_keypoints[key1][...]).half().to(device)
                descriptors1 = torch.from_numpy(f_descriptors[key1][...]).half().to(device)
                keypoints2 = torch.from_numpy(f_keypoints[key2][...]).half().to(device)
                descriptors2 = torch.from_numpy(f_descriptors[key2][...]).half().to(device)

                with torch.inference_mode():
                    _, indices = matcher(
                        descriptors1,
                        descriptors2,
                        KF.laf_from_center_scale_ori(keypoints1[None]),
                        KF.laf_from_center_scale_ori(keypoints2[None]),
                    )

                keypoints1 = keypoints1[0].cpu().numpy()
                keypoints2 = keypoints2[0].cpu().numpy()
                indices = indices.cpu().numpy()

                group  = f_matches.require_group(key1)
                group.create_dataset(key2, data=indices)

    return

if __name__ == "__main__":

    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, default="../output/debug")
    parser.add_argument("--draw_results", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = list(args.image_dir.glob("*"))

    feature_dir = args.output_dir / ".features"

    get_image_pairs_args = {
        "model_name": "facebook/dinov2-base",
        "similarity_threshold": 0.25,
        "tolerance": 1000,
        "min_matches": 20,
        "exhaustive_if_less": 20,
        "p": 2.0,
    }

    index_pairs, _ = get_image_pairs(
        image_paths=image_paths,
        device=device,
        **get_image_pairs_args,
    )

    feature_matching_aliked_lightglue(
        image_paths=image_paths,
        index_pairs=index_pairs,
        feature_dir=feature_dir,
        device=device,
    )

    if args.draw_results:
        from draw_matching import draw_matching

        with h5py.File(feature_dir / "keypoints.h5", mode="r") as f_keypoints, \
            h5py.File(feature_dir / "matches.h5", mode="r") as f_matches:

            for key1 in tqdm(f_matches.keys(), desc="Drawing matches"):

                group = f_matches[key1]
                keypoints1 = f_keypoints[key1][...]

                for key2 in group.keys():

                    keypoints2 = f_keypoints[key2][...]
                    matches = group[key2][...]

                    # Verify using Two-View Geometry
                    mkpts1 = keypoints1[matches[:, 0]]
                    mkpts2 = keypoints2[matches[:, 1]]

                    try:
                        Fm, inliers = cv2.findFundamentalMat(
                            mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
                        inliers = inliers > 0
                    except:
                        continue

                    title = f"{key1[:-4]}-{key2[:-4]}"
                    title += f"\n # of matches: {matches.shape[0]}, # of inliers: {inliers.sum()}"
                    draw_matching(
                        args.image_dir,
                        args.output_dir / f"{key1[:-4]}-{key2[:-4]}.png",
                        key1[:-4], key2[:-4], "JPG",
                        mkpts1, mkpts2, inliers,
                        title=title,
                    )