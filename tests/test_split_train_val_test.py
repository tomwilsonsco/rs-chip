import pytest
import tempfile
from pathlib import Path
import rasterio as rio
import numpy as np
from rschip.split_train_val_test import DatasetSplitter


@pytest.fixture
def setup_test_data():
    with tempfile.TemporaryDirectory() as test_dir:
        image_dir = Path(test_dir) / "images"
        mask_dir = Path(test_dir) / "masks"
        output_dir = Path(test_dir) / "output"
        image_dir.mkdir()
        mask_dir.mkdir()
        output_dir.mkdir()

        # Create dummy files
        for i in range(10):
            profile = {
                "driver": "GTiff",
                "height": 1,
                "width": 1,
                "count": 1,
                "dtype": "uint8",
                "transform": rio.transform.from_origin(0, 0, 1, 1),
                "crs": "EPSG:27700",
            }
            with rio.open(image_dir / f"test_{i}.tif", "w", **profile) as dst:
                dst.write(np.zeros((1, 1, 1), dtype="uint8"))

            mask_data = (
                np.ones((1, 1, 1), dtype="uint8")
                if i < 8
                else np.zeros((1, 1, 1), dtype="uint8")
            )
            with rio.open(mask_dir / f"test_{i}.tif", "w", **profile) as dst:
                dst.write(mask_data)

        yield {"image_dir": image_dir, "mask_dir": mask_dir, "output_dir": output_dir}


def test_invalid_ratios(setup_test_data):
    dirs = setup_test_data
    with pytest.raises(ValueError):
        DatasetSplitter(
            dirs["image_dir"],
            dirs["mask_dir"],
            dirs["output_dir"],
            train_ratio=0.5,
            val_ratio=0.5,
            test_ratio=0.1,
        )
    with pytest.raises(ValueError):
        DatasetSplitter(
            dirs["image_dir"],
            dirs["mask_dir"],
            dirs["output_dir"],
            train_ratio=0,
            val_ratio=0.9,
            test_ratio=0.1,
        )


def test_directory_creation(setup_test_data):
    dirs = setup_test_data
    splitter = DatasetSplitter(
        dirs["image_dir"], dirs["mask_dir"], dirs["output_dir"], seed=42
    )
    splitter.split()
    dataset_dir = dirs["output_dir"] / "dataset"
    assert dataset_dir.exists()
    assert (dataset_dir / "images" / "train").exists()
    assert (dataset_dir / "images" / "val").exists()
    assert (dataset_dir / "images" / "test").exists()
    assert (dataset_dir / "masks" / "train").exists()
    assert (dataset_dir / "masks" / "val").exists()
    assert (dataset_dir / "masks" / "test").exists()

    train_images = list(
        (dirs["output_dir"] / "dataset" / "images" / "train").glob("*.tif")
    )
    val_images = list((dirs["output_dir"] / "dataset" / "images" / "val").glob("*.tif"))
    test_images = list(
        (dirs["output_dir"] / "dataset" / "images" / "test").glob("*.tif")
    )

    assert len(train_images) == 7
    assert len(val_images) == 1
    assert len(test_images) == 0


def test_filter_background_only(setup_test_data):
    dirs = setup_test_data
    # 8 non-background files, 2 background files. The background check should run automatically.
    # The split is on 8 files, 6 train, 2 val
    splitter = DatasetSplitter(
        dirs["image_dir"],
        dirs["mask_dir"],
        dirs["output_dir"],
        train_ratio=0.75,
        val_ratio=0.25,
        test_ratio=0.0,
        seed=42,
    )
    splitter.split()

    train_images = list(
        (dirs["output_dir"] / "dataset" / "images" / "train").glob("*.tif")
    )
    val_images = list((dirs["output_dir"] / "dataset" / "images" / "val").glob("*.tif"))
    assert len(train_images) == 6
    assert len(val_images) == 2
    assert not (dirs["output_dir"] / "dataset" / "images" / "test").exists()
