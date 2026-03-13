import warnings
import shutil
import random
import multiprocessing
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from rschip.check_background import CheckBackgroundOnly


def _copy_worker(args):
    """Worker function to copy a single image-mask pair."""
    img_path, mask_path, img_dest, mask_dest = args
    shutil.copy(img_path, img_dest)
    shutil.copy(mask_path, mask_dest)


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
        filter_background_only=True,
        use_multiprocessing=True,
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
            filter_background_only (bool): If True, uses the CheckBackgroundOnly class to filter out
            background-only image/mask pairs before copying the remaining files into the dataset.
            Defaults to True.
            use_multiprocessing (bool): Whether to use multiprocessing for copying files and background-only check.
            Defaults to True.

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
        self.filter_background_only = filter_background_only
        self.use_multiprocessing = use_multiprocessing

        if self.dataset_dir.exists():
            raise ValueError(
                f"Output directory '{self.dataset_dir}' already exists. "
                "Please remove it or choose a different output directory."
            )

        # Ensure each ratio is a valid proportion in [0, 1]
        for name, value in (
            ("train_ratio", train_ratio),
            ("val_ratio", val_ratio),
            ("test_ratio", test_ratio),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1 (received {value}).")

        # Ensure the ratios collectively sum to 1 (within a small tolerance)
        if not (0.999 < train_ratio + val_ratio + test_ratio < 1.001):
            raise ValueError("The sum of train, val, and test ratios must be 1.")

        # Train and validation ratios must be strictly greater than 0
        if train_ratio <= 0 or val_ratio <= 0:
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
        all_images = sorted(self.image_dir.glob("*.tif"), key=lambda p: p.name)
        existing_masks = {p.name for p in self.mask_dir.glob("*.tif")}  # one-time scan
        file_pairs = []
        for img_path in all_images:
            if img_path.name in existing_masks:
                file_pairs.append((img_path, self.mask_dir / img_path.name))
            else:
                warnings.warn(
                    f"No equivalent mask found for {img_path}",
                    UserWarning,
                    stacklevel=2,
                )
        return file_pairs

    def _filter_background_only(self, file_pairs):
        """Filters out file pairs that are background only."""
        if not self.filter_background_only:
            print("Skipping background check.")
            return file_pairs

        background_csv = self.mask_dir / "background_only_check.csv"
        if not background_csv.exists():
            print(
                f"'{background_csv}' not found. Running check for background only chips"
            )
            checker = CheckBackgroundOnly(use_multiprocessing=self.use_multiprocessing)
            df = checker.check_background_chips(str(self.mask_dir), str(self.image_dir))
        else:
            df = pd.read_csv(background_csv)

        background_files = set(
            Path(f).name for f in df[df["is_background_only"]]["mask_file"]
        )

        filtered_pairs = [
            (img_path, mask_path)
            for img_path, mask_path in file_pairs
            if mask_path.name not in background_files
        ]

        return filtered_pairs

    def split(self):
        """Splits the dataset."""
        file_pairs = self._get_file_pairs()

        if not file_pairs:
            raise FileNotFoundError(
                "No image-mask pairs found in the specified input directories."
            )

        file_pairs = self._filter_background_only(file_pairs)

        total_files = len(file_pairs)
        if total_files == 0:
            raise ValueError(
                "No valid images found after filtering background-only masks."
            )

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

        self._copy_files(
            train_files, self.train_images_dir, self.train_masks_dir, "train"
        )
        self._copy_files(val_files, self.val_images_dir, self.val_masks_dir, "val")
        if self.test_ratio > 0:
            self._copy_files(
                test_files, self.test_images_dir, self.test_masks_dir, "test"
            )

        print("Dataset splitting complete.")

    def _copy_files(self, files, img_dest, mask_dest, set_name: str):
        """Copies a list of file pairs to the destination directories."""
        if not files:
            return

        desc = f"Copying {set_name} files"
        if self.use_multiprocessing:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
            print(f"Copying {len(files)} {set_name} files using {num_cores} cores.")

            task_iter = (
                (img_path, mask_path, img_dest, mask_dest)
                for img_path, mask_path in files
            )
            with multiprocessing.Pool(num_cores) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_copy_worker, task_iter),
                    total=len(files),
                    desc=desc,
                ):
                    pass
        else:
            print(f"Copying {len(files)} {set_name} files sequentially.")
            for img_path, mask_path in tqdm(files, desc=desc):
                shutil.copy(img_path, img_dest)
                shutil.copy(mask_path, mask_dest)
