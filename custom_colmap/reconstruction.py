import pycolmap

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, default="../output/debug")
    args = parser.parse_args()

    database_path = args.output_dir / "distorted" / "database.db"
    output_path = args.output_dir / "distorted" / "sparse"
    output_path.mkdir(parents=True, exist_ok=True)

    match_exhaustive_options = {
        "gpu_index": "0",
        "guided_matching": True,
    }
    pycolmap.match_exhaustive(
        database_path=database_path,
        sift_options=pycolmap.SiftMatchingOptions(**match_exhaustive_options),
    )

    mapper_options = {
        "ba_local_max_num_iterations": 40,
        "ba_local_max_refinements": 3,
        "ba_global_max_num_iterations": 100,
    }
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=args.image_dir,
        output_path=output_path,
        options=pycolmap.IncrementalPipelineOptions(**mapper_options),
    )

    if isinstance(maps, dict):
        for idx, rec in maps.items():
            print(idx, rec.summary())
