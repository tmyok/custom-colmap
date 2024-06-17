import argparse
import gc
import pycolmap
import shutil
import torch

from pathlib import Path

from custom_colmap.feature_extraction import detect_keypoints_ALIKED
from custom_colmap.get_image_pairs import get_image_pairs
from custom_colmap.feature_matching import feature_matching_aliked_lightglue
from custom_colmap.import_into_colmap import import_into_colmap

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Colmap converter")
    parser.add_argument("--source_path", "-s", required=True, type=Path)
    parser.add_argument("--camera", default="opencv", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = args.source_path / "input"
    image_paths = list(image_dir.glob("*"))
    feature_dir = args.source_path / ".features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    pycolmap.set_random_seed(42)

    ## Feature extraction
    detect_keypoints_ALIKED(
        image_paths=image_paths,
        feature_dir=feature_dir,
        device=device,
        resize_to=2048,
    )
    torch.cuda.empty_cache()
    gc.collect()

    ## Get image pairs
    index_pairs, _ = get_image_pairs(
        image_paths=image_paths,
        device=device,
        similarity_threshold=0.25,
    )
    torch.cuda.empty_cache()
    gc.collect()

    ## Feature matching
    feature_matching_aliked_lightglue(
        image_paths=image_paths,
        index_pairs=index_pairs,
        feature_dir=feature_dir,
        device=device,
    )
    torch.cuda.empty_cache()
    gc.collect()

    ## Import into Colmap
    distorted_dir = args.source_path / "distorted"
    distorted_dir.mkdir(parents=True, exist_ok=True)

    database_path = distorted_dir / "database.db"

    if database_path.exists():
        database_path.unlink()
    import_into_colmap(
        image_dir=image_dir,
        feature_dir=feature_dir,
        database_path=database_path,
        camera_model=args.camera,
        single_camera=True,
    )
    shutil.rmtree(feature_dir)
    gc.collect()

    match_exhaustive_options = {
        "gpu_index": "0",
        "guided_matching": True,
    }
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options=pycolmap.SiftMatchingOptions(**match_exhaustive_options),
    )
    torch.cuda.empty_cache()
    gc.collect()

    ## Bundle adjustment
    output_path = distorted_dir / "sparse"
    output_path.mkdir(parents=True, exist_ok=True)

    mapper_options = {
        "ba_local_max_num_iterations": 40,
        "ba_local_max_refinements": 3,
        "ba_global_max_num_iterations": 100,
    }
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=output_path,
        options=pycolmap.IncrementalPipelineOptions(**mapper_options),
    )
    gc.collect()

    ### Image undistortion
    ## We need to undistort our images into ideal pinhole intrinsics.
    distorted_dir = args.source_path / "distorted"
    output_path = distorted_dir / "sparse"
    pycolmap.undistort_images(
        output_path=args.source_path,
        input_path=output_path / "0",
        image_path=image_dir,
        output_type="COLMAP",
    )
    gc.collect()

    # Copy each file from the source directory to the destination directory
    source_dir = args.source_path / "sparse"
    destination_dir = args.source_path / "sparse" / "0"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for source_file in [p for p in source_dir.iterdir() if p.is_file()]:
        shutil.move(source_file, destination_dir)

    print("Done.")