import itertools
import kornia as K
import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel

def embed_images(
    image_paths,
    model_name,
    device,
):

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)

    embeddings = []

    for i, path in tqdm(enumerate(image_paths), desc="Global descriptors", dynamic_ncols=True, total=len(image_paths)):
        image = K.io.load_image(path, K.io.ImageLoadType.RGB32, device=device)[None, ...]

        with torch.inference_mode():
            inputs = processor(images=image, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs) # last_hidden_state and pooled

            # Max pooling over all the hidden states but the first (starting token)
            # To obtain a tensor of shape [1, output_dim]
            # We normalize so that distances are computed in a better fashion later
            embedding = F.normalize(outputs.last_hidden_state[:,1:].max(dim=1)[0], dim=-1, p=2)

        embeddings.append(embedding.detach().cpu())
    return torch.cat(embeddings, dim=0)

def get_pairs_exhaustive(lst):
    return list(itertools.combinations(range(len(lst)), 2))

def get_image_pairs(
    image_paths,
    device,
    model_name = "facebook/dinov2-base",
    similarity_threshold = 0.25,
    tolerance = 1000,
    min_matches = 20,
    exhaustive_if_less = 20,
    p = 2.0,
):
    if len(image_paths) <= exhaustive_if_less:
        pairs = get_pairs_exhaustive(image_paths)
        distances = np.zeros((len(image_paths), len(image_paths)))
        return pairs, distances

    embeddings = embed_images(image_paths, model_name, device=device)
    distances = torch.cdist(embeddings, embeddings, p=p)

    mask = distances <= similarity_threshold
    image_indices = np.arange(len(image_paths))

    matches = []
    distance_list = []
    for current_image_index in range(len(image_paths)):
        mask_row = mask[current_image_index]
        indices_to_match = image_indices[mask_row]

        if len(indices_to_match) < min_matches:
            indices_to_match = np.argsort(distances[current_image_index].numpy())[:min_matches]

        for other_image_index in indices_to_match:
            if other_image_index == current_image_index:
                continue

            if distances[current_image_index, other_image_index] < tolerance:
                matches.append(tuple(sorted((current_image_index, other_image_index.item()))))
                distance_list.append(distances[current_image_index, other_image_index].item())

    pairs = sorted(list(set(matches)))
    distances = np.array(distance_list)

    return pairs, distances

if __name__ == "__main__":

    import argparse
    import cv2
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=pathlib.Path, required=True)
    parser.add_argument("--output_dir", type=pathlib.Path, default="../output/debug")
    parser.add_argument("--draw_results", action="store_true")
    args = parser.parse_args()

    image_paths = list(args.image_dir.glob("*"))
    # random sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    get_image_pairs_args = {
        "model_name": "facebook/dinov2-base",
        "similarity_threshold": 0.25,
        "tolerance": 1000,
        "min_matches": 20,
        "exhaustive_if_less": 20,
        "p": 2.0,
    }

    index_pairs, distances = get_image_pairs(
        image_paths=image_paths,
        device=device,
        **get_image_pairs_args,
    )

    if args.draw_results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        for index_pair, distance in tqdm(zip(index_pairs, distances), desc="Drawing pairs", dynamic_ncols=True, total=len(index_pairs)):

            key1 = image_paths[index_pair[0]].stem
            key2 = image_paths[index_pair[1]].stem

            image1 = cv2.imread(str(image_paths[index_pair[0]]))
            image2 = cv2.imread(str(image_paths[index_pair[1]]))


            fig, ax = plt.subplots(1, 2, figsize=(8, 6))
            ax[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
            ax[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

            title = f"Distance: {distance:.2f}"
            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(args.output_dir / f"{key1}-{key2}_dist={distance:.2f}.png")
            plt.clf()
            plt.close()