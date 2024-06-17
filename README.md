# Custom COLMAP

This repository demonstrates how to incorporate and use deep-learning based local features and matchers in COLMAP. Compared to the default SIFT features of COLMAP, it can provide denser and higher quality image matching.

## Usage

`convert.py` is for preparing input image data for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/). Place the images you wish to use in the `<location>/input` directory.

```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
To run the Custom COLMAP code, which serves as an alternative to the original implementation of [convert.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py), execute the following command:
```
python3 convert.py --source_path <location>
```

## Docker

Run the Docker container (if necessary).
```
sh docker_container.sh
```