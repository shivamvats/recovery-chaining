from argparse import ArgumentParser
from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from segment_anything import SamPredictor, sam_model_registry

path_to_sam_model = "data/sam_vit_h_4b8939.pth"


def predict_segments(img, prompt):
    sam = sam_model_registry["<model_type>"](checkpoint=path_to_sam_model)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, _, _ = predictor.predict(prompt)

def main(args):
    predict_segments(img, prompt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", "-o", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    args = parser.parse_args()
    main(args)
