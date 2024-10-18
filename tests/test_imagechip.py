import shutil
import pytest
import numpy as np
import rasterio as rio
from pathlib import Path
import tempfile
from rschip import ImageChip


@pytest.fixture(scope="function")
def setup_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def chip_image_run(
    output_path,
    input_image_path="tests/data/test_img.tif",
    pixel_dimensions=128,
    offset=64,
    standard_scale=True,
    sample_size=10000,
    scaler_source=None,
    use_multiprocessing=True,
    output_format="tif",
    max_batch_size=10,
):
    image_chip = ImageChip(
        input_image_path=input_image_path,
        output_path=output_path,
        pixel_dimensions=pixel_dimensions,
        offset=offset,
        standard_scale=standard_scale,
        sample_size=sample_size,
        scaler_source=scaler_source,
        use_multiprocessing=use_multiprocessing,
        output_format=output_format,
        max_batch_size=max_batch_size,
    )
    image_chip.chip_image()


def load_npz(npz_file_path):
    with np.load(npz_file_path) as data:
        test_key = data.files[0]
        arr = data[test_key]
    return test_key, arr


def load_tif(tif_file_path):
    with rio.open(tif_file_path) as f:
        arr = f.read()
        prof = f.profile
    return arr, prof


def npz_files_to_list(out_dir):
    array_list = []
    for file_path in Path(out_dir).glob("*.npz"):
        with np.load(file_path) as data:
            for array_name in data.files:
                array_list.append(data[array_name])
    return array_list


def tif_files_to_list(out_dir):
    return list(Path(out_dir).glob("*.tif"))


def test_image_chip(setup_output_dir):
    out_dir = setup_output_dir
    # Test chipping with TIFF output
    chip_image_run(output_path=out_dir)

    # Verify that TIFF files were created
    tif_files = list(out_dir.glob("*.tif"))
    assert len(tif_files) > 0, "No TIFF files were created."

    # Test chipping with NPZ output using the same scaler file
    chip_image_run(output_path=out_dir, output_format="npz")

    # Verify that NPZ files were created
    npz_files = list(out_dir.glob("*.npz"))
    assert len(npz_files) > 0, "No NPZ files were created."


def test_array_equality(setup_output_dir):
    out_dir = setup_output_dir
    scaler_fp = f"{out_dir}/test_img_10000.pkl"
    chip_image_run(output_path=out_dir, output_format="tif")
    chip_image_run(output_path=out_dir, output_format="npz", scaler_source=scaler_fp)

    # Load one NPZ and corresponding TIFF file to compare arrays
    test_key, npz_arr = load_npz(out_dir / "batch_0.npz")
    tif_arr, _ = load_tif(f"{out_dir}/{test_key}.tif")

    # Compare arrays
    assert np.array_equal(tif_arr, npz_arr), "Arrays are not equal."


def test_image_and_array_count(setup_output_dir):
    out_dir = setup_output_dir

    chip_image_run(output_path=out_dir, output_format="tif")
    chip_image_run(output_path=out_dir, output_format="npz")

    # Compare the number of TIFF images and NPZ arrays
    npz_array_count = len(npz_files_to_list(out_dir))
    tif_file_count = len(tif_files_to_list(out_dir))
    assert npz_array_count == tif_file_count, "Mismatch in number of arrays and images."


def test_scaler_functionality():
    sample_array = np.random.rand(3, 128, 128) * 100
    scaler_dict = {
        0: {"mean": 50, "std": 10},
        1: {"mean": 60, "std": 15},
        2: {"mean": 40, "std": 20},
    }
    scaled_array = ImageChip.apply_scaler(sample_array, scaler_dict)
    unscaled_array = ImageChip.unapply_scaler(scaled_array, scaler_dict)
    assert np.allclose(
        sample_array, unscaled_array, atol=1e-1
    ), "Scaling/unscaling mismatch"


def test_large_window_image(setup_output_dir):
    out_dir = setup_output_dir
    chip_image_run(
        output_path=out_dir, output_format="tif", pixel_dimensions=512, offset=256
    )
    chip_image_run(
        output_path=out_dir, output_format="npz", pixel_dimensions=512, offset=256
    )

    # Verify that at least one file was created
    tif_files = tif_files_to_list(out_dir)
    assert len(tif_files) > 0, "No TIFF files were created for large window."


def test_multiprocessor_not(setup_output_dir):
    out_dir = setup_output_dir
    chip_image_run(output_path=out_dir, use_multiprocessing=True)
    mp_files = tif_files_to_list(out_dir)

    shutil.rmtree(out_dir)
    out_dir = setup_output_dir
    chip_image_run(output_path=out_dir, use_multiprocessing=False)
    sp_files = tif_files_to_list(out_dir)

    assert (
        mp_files == sp_files
    ), "multiprocessing and single processing have different results"


def test_tile_count(setup_output_dir):
    out_dir = setup_output_dir
    input_image_path = "tests/data/test_img.tif"
    pixel_dimensions = 128
    offset = 64

    # Read the input image to get its dimensions
    with rio.open(input_image_path) as src:
        img_height = src.height
        img_width = src.width

    # Calculate expected number of tiles
    expected_tiles_x = (img_width + offset - 1) // offset
    expected_tiles_y = (img_height + offset - 1) // offset
    expected_tile_count = expected_tiles_x * expected_tiles_y

    # Run the chipping process
    chip_image_run(
        output_path=out_dir,
        input_image_path=input_image_path,
        pixel_dimensions=pixel_dimensions,
        offset=offset,
        output_format="tif",
    )

    # Verify that the expected number of tiles were created
    tif_files = tif_files_to_list(out_dir)
    assert (
        len(tif_files) == expected_tile_count
    ), f"Expected {expected_tile_count} tiles, but found {len(tif_files)}."


if __name__ == "__main__":
    pytest.main()
