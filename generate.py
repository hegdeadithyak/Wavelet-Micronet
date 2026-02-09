import os
import argparse
import numpy as np
from PIL import Image


PURPLE = np.array([88, 0, 110], dtype=np.uint8)
TEAL   = np.array([45, 150, 140], dtype=np.uint8)


def generate_clean_wafer(size=512):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = PURPLE

    cx, cy = size // 2, size // 2
    outer_radius = size // 2 - 10

    for y in range(size):
        for x in range(size):
            dx = x - cx
            dy = y - cy
            if (dx*dx + dy*dy) ** 0.5 <= outer_radius:
                img[y, x] = TEAL

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, required=True,
                        help="Number of images to generate")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--size", type=int, default=512,
                        help="Image size (default: 512)")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.num_images):
        img = generate_clean_wafer(args.size)
        path = os.path.join(args.out_dir, f"wafer_{i:05d}.png")
        Image.fromarray(img).save(path)

    print(f"Generated {args.num_images} images in {args.out_dir}")


if __name__ == "__main__":
    main()
