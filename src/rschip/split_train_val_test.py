import shutil
import random
from pathlib import Path
import pandas as pd
from rschip.check_background import CheckBackgroundOnly


class DatasetSplitter:
    """
    Splits a dataset of images and masks into training, validation, and testing sets.

    The class creates a 'dataset' directory in the specified output directory with subdirectories
    for 'images' and 'masks', each containing 'train', 'val', and optional 'test' folders.

    It checks for consistent pairs of image and mask files, handles background-only masks,
    and splits the data according to specified ratios.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        output_dir,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=None,
        run_background_check=True,
    ):
        """
        Initializes the DatasetSplitter.

        Args:
            image_dir (str): Path to the directory containing the images.
            mask_dir (str): Path to the directory containing the masks.
            output_dir (str): Path to the root directory where the 'dataset' directory will be created.
            train_ratio (float): The proportion of the data to be used for training. Must be > 0.
            val_ratio (float): The proportion of the data to be used for validation. Must be > 0.
            test_ratio (float): The proportion of the data to be used for testing. Can be 0.
            seed (int, optional): Seed for the random number generator for reproducible shuffling. Defaults to None.
            run_background_check (bool): If True, the process will only move files without background
            by using the CheckBackgroundOnly class. Defaults to True.

        Raises:
            ValueError: If the output directory/dataset already exists, if split ratios do not sum to 1,
                        or if train or validation ratios are 0.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.output_dir = Path(output_dir)
        self.dataset_dir = self.output_dir / "dataset"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.run_background_check = run_background_check

        if self.dataset_dir.exists():
            raise ValueError(f"Output directory '{self.dataset_dir}' already exists. \
                Please remove it or choose a different output directory.")

        if not (0.999 < train_ratio + val_ratio + test_ratio < 1.001):
            raise ValueError("The sum of train, val, and test ratios must be 1.")

        if train_ratio == 0 or val_ratio == 0:
            raise ValueError("Train and validation ratios must be greater than 0.")

        self.images_out_dir = self.dataset_dir / "images"
        self.masks_out_dir = self.dataset_dir / "masks"
        self.train_images_dir = self.images_out_dir / "train"
        self.val_images_dir = self.images_out_dir / "val"
        self.test_images_dir = self.images_out_dir / "test"
        self.train_masks_dir = self.masks_out_dir / "train"
        self.val_masks_dir = self.masks_out_dir / "val"
        self.test_masks_dir = self.masks_out_dir / "test"

    def _create_dirs(self):
        """Creates the output directories."""
        self.dataset_dir.mkdir(parents=True)
        self.images_out_dir.mkdir()
        self.masks_out_dir.mkdir()
        self.train_images_dir.mkdir()
        self.val_images_dir.mkdir()
        self.train_masks_dir.mkdir()
        self.val_masks_dir.mkdir()
        if self.test_ratio > 0:
            self.test_images_dir.mkdir()
            self.test_masks_dir.mkdir()

    def _get_file_pairs(self):
        """
        Finds pairs of corresponding images and masks.
        Assumes that for each image file, there is a mask file with the same name.
        """
        all_images = list(self.image_dir.glob("*.tif"))
        file_pairs = []
        for img_path in all_images:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                file_pairs.append((img_path, mask_path))
        return file_pairs

    def _filter_background_only(self, file_pairs):
        """Filters out file pairs that are background only."""
        if not self.run_background_check:
            print("Skipping background check.")
            return file_pairs

        background_csv = self.mask_dir / "background_only_check.csv"
        if not background_csv.exists():
            print(
                f"'{background_csv}' not found. Running check for background only chips"
            )
            checker = CheckBackgroundOnly()
            checker.check_background_chips(str(self.mask_dir), str(self.image_dir))

        df = pd.read_csv(background_csv)
        background_files = df[df["is_background_only"]]["mask_file"].tolist()
        background_files = [Path(f).name for f in background_files]

        filtered_pairs = []
        for img_path, mask_path in file_pairs:
            if mask_path.name not in background_files:
                filtered_pairs.append((img_path, mask_path))

        return filtered_pairs

    def split(self):
        """Splits the dataset."""
        file_pairs = self._get_file_pairs()

        if not file_pairs:
            print("Error: No image-mask pairs found.")
            return

        file_pairs = self._filter_background_only(file_pairs)

        total_files = len(file_pairs)
        if total_files == 0:
            print("Error: No valid images found after filtering.")
            return

        self._create_dirs()

        print(f"Found {total_files} valid image-mask pairs for splitting.")

        if self.seed is not None:
            rng = random.Random(self.seed)
            rng.shuffle(file_pairs)
        else:
            random.shuffle(file_pairs)

        test_count = int(total_files * self.test_ratio)
        val_count = int(total_files * self.val_ratio)

        test_files = file_pairs[:test_count]
        train_start = test_count + val_count
        val_files = file_pairs[test_count:train_start]
        train_files = file_pairs[train_start:]

        self._copy_files(train_files, self.train_images_dir, self.train_masks_dir)
        self._copy_files(val_files, self.val_images_dir, self.val_masks_dir)
        if self.test_ratio > 0:
            self._copy_files(test_files, self.test_images_dir, self.test_masks_dir)

        print("Dataset splitting complete.")

    def _copy_files(self, files, img_dest, mask_dest):
        """Copies a list of file pairs to the destination directories."""
        for img_path, mask_path in files:
            shutil.copy(img_path, img_dest)
            shutil.copy(mask_path, mask_dest)
