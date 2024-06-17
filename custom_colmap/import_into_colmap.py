from .colmap_util.database import COLMAPDatabase
from .colmap_util.h5_to_db import add_keypoints, add_matches

def import_into_colmap(
    image_dir,
    feature_dir,
    database_path,
    camera_model,
    focal_length_dict=None,
    single_camera=False,
):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(
        db,
        feature_dir,
        image_dir,
        camera_model,
        single_camera,
        focal_length_dict,
    )
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )
    db.commit()

    return

if __name__ == "__main__":

    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, default="../output/debug")

    args = parser.parse_args()

    feature_dir = args.output_dir / ".features"
    database_path = args.output_dir / "database.db"
    if database_path.exists():
        database_path.unlink()

    import_into_colmap(
        args.image_dir,
        feature_dir,
        database_path,
       camera_model="opencv",
        single_camera=True,
    )