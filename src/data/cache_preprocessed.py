import contextlib
import os
from collections import Counter
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import RAW_DIR, PROCESSED_DIR
from src.data.preprocessing import preprocess_pipeline

INPUT_DIR = RAW_DIR / "train_images"
OUTPUT_DIR = PROCESSED_DIR / "images_processed"

def process_single_image(img_id: str) -> str:
    """
    Applies the preprocessing pipeline to a single image and saves it to disk.

    Args:
        img_id (str): The id of the image.

    Returns:
        str: Status of the processing request ("success", "failure", "skipped" or "missing").
    """
    cv2.setNumThreads(1)

    img_name = f"{img_id}.png"
    input_path = INPUT_DIR / img_name
    output_path = OUTPUT_DIR / img_name

    if output_path.exists():
        return "skipped"

    if not input_path.exists():
        return "missing"

    temp_path = OUTPUT_DIR / f"{img_id}_temp.png"
    try:
        processed_image_rgb = preprocess_pipeline(str(input_path))
        processed_image_bgr = cv2.cvtColor(processed_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(temp_path), processed_image_bgr)
        os.replace(str(temp_path), str(output_path))
        return "success"
    except Exception as e:
        print(f"Error processing image {img_id}.png: {e}")
        if temp_path.exists():
            with contextlib.suppress(Exception):
                temp_path.unlink()
        return "failure"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DIR / "train.csv")
    num_processes = min(4, cpu_count())

    with Pool(processes=num_processes) as pool:
        result = list(tqdm(pool.imap(process_single_image, df['id_code']), total=len(df)))

    cnt = Counter(result)

    processed_files_count = len(list(OUTPUT_DIR.glob('*.png')))
    print(f"Total processed files on disk: {processed_files_count}")
    EXPECTED_COUNT = len(df)
    if processed_files_count != EXPECTED_COUNT:
        print(f"Number of processed files ({processed_files_count}) does not match expected value ({EXPECTED_COUNT})!")

    print(f"Success: {cnt['success']}")
    print(f"Failure: {cnt['failure']}")
    print(f"Skipped: {cnt['skipped']}")
    print(f"Missing: {cnt['missing']}")

    # Verify cache integrity
    print("\nIntegration test:")
    sample_id = df.loc[0, "id_code"]
    original_image_path = INPUT_DIR / f"{sample_id}.png"
    cached_image_path = OUTPUT_DIR / f"{sample_id}.png"

    if cached_image_path.exists():
        expected_image_rgb = preprocess_pipeline(str(original_image_path))
        cached_image_bgr = cv2.imread(str(cached_image_path))
        cached_image_rgb = cv2.cvtColor(cached_image_bgr, cv2.COLOR_BGR2RGB)

        if np.array_equal(expected_image_rgb, cached_image_rgb):
            print(f"[Verification Passed] Cached image '{sample_id}' matches the preprocessing pipeline output.")
        else:
            print(f"[Verification Failed] Cached image '{sample_id}' differs from the preprocessing pipeline output!")
    else:
        print(f"[Verification Skipped] Sample image '{sample_id}' not found in cache.")