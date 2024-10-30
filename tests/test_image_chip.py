import shutil
import pytest
import numpy as np
import rasterio as rio
from pathlib import Path
import tempfile
from rschip import ImageChip
import pickle
import re


@pytest.fixture(scope="function")
def setup_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def chip_image_run(
    output_path,
    input_image_path="tests/data/test_img.tif",
    pixel_dimensions=128,
    offset=64,
    use_multiprocessing=True,
    output_format="tif",
    max_batch_size=10,
    scaler=False,
    normaliser=False,
):
    image_chip = ImageChip(
        input_image_path=input_image_path,
        output_path=output_path,
        pixel_dimensions=pixel_dimensions,
        offset=offset,
        use_multiprocessing=use_multiprocessing,
        output_format=output_format,
        max_batch_size=max_batch_size,
    )
    if scaler:
        image_chip.set_scaler()
    if normaliser:
        image_chip.set_normaliser(min_val=1000, max_val=3000)

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

    # Test chipping with NPZ output
    chip_image_run(output_path=out_dir, output_format="npz")

    # Verify that NPZ files were created
    npz_files = list(out_dir.glob("*.npz"))
    assert len(npz_files) > 0, "No NPZ files were created."


def test_array_equality(setup_output_dir):
    out_dir = setup_output_dir
    chip_image_run(output_path=out_dir, output_format="tif")
    chip_image_run(output_path=out_dir, output_format="npz")

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


def test_normalising(setup_output_dir):
    out_dir = setup_output_dir

    chip_image_run(output_path=out_dir, output_format="tif", normaliser=True)

    with rio.open(tif_files_to_list(out_dir)[0]) as f:
        test_array = f.read()
    max_val = np.max(test_array)
    assert max_val <= 1, "normalising not worked as values > 1 in array."


def test_standard_scaling(setup_output_dir):
    out_dir = setup_output_dir

    chip_image_run(output_path=out_dir, output_format="tif", scaler=True)

    with rio.open(tif_files_to_list(out_dir)[0]) as f:
        test_array = f.read()
    mean_val = np.mean(test_array)
    assert mean_val <= 10, "standard scaling not worked as mean > 10 in array."


def test_validate_normaliser_inputs_valid_float():
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path="tmp"
    )
    result = image_chip._validate_normaliser_inputs(5.0, "min_val")
    assert result == [5.0, 5.0]


def test_validate_normaliser_inputs_valid_list():
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path="tmp"
    )
    value = [1.0, 2.0]
    result = image_chip._validate_normaliser_inputs(value, "min_val")
    assert result == value


def test_validate_normaliser_inputs_invalid_list_length():
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path="tmp"
    )
    value = [1.0]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "min_val list [1.0] must be the same length as the number of bands in the image (2)."
        ),
    ):
        image_chip._validate_normaliser_inputs(value, "min_val")


def test_validate_normaliser_inputs_invalid_list_element():
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path="tmp"
    )
    value = [1.0, "a"]
    with pytest.raises(
        ValueError, match="min_val list .* must only contain integer or float numbers"
    ):
        image_chip._validate_normaliser_inputs(value, "min_val")


def test_validate_normaliser_inputs_invalid_type():
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path="tmp"
    )
    value = {"key": "value"}
    with pytest.raises(ValueError, match="min_val must be either list, float, or int"):
        image_chip._validate_normaliser_inputs(value, "min_val")


def test_set_normaliser(setup_output_dir):
    out_dir = setup_output_dir
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path=out_dir
    )
    image_chip.set_normaliser(min_val=1.0, max_val=5.0)
    assert image_chip.normaliser == {"min_val": [1.0, 1.0], "max_val": [5.0, 5.0]}


def test_pickle_default_normaliser(setup_output_dir):
    out_dir = setup_output_dir
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path=out_dir
    )
    image_chip.set_normaliser(min_val=1.0, max_val=5.0)
    pickle_file_name = f"{image_chip.input_image_path.stem}_normaliser.pkl"
    output_dir = Path(image_chip.output_path)
    pickle_file_path = output_dir / pickle_file_name
    with open(pickle_file_path, "rb") as f:
        file_normaliser = pickle.load(f)
    assert image_chip.normaliser == file_normaliser


def test_pickle_custom_normaliser(setup_output_dir):
    out_dir = setup_output_dir
    custom_pickle_dir = out_dir / "normaliser_test.pkl"
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path=out_dir
    )
    image_chip.set_normaliser(min_val=1.0, max_val=5.0, write_path=custom_pickle_dir)
    with open(custom_pickle_dir, "rb") as f:
        file_normaliser = pickle.load(f)
    assert image_chip.normaliser == file_normaliser


def test_pickle_default_scaler(setup_output_dir):
    out_dir = setup_output_dir
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path=out_dir
    )
    image_chip.set_scaler()
    pickle_file_name = f"{image_chip.input_image_path.stem}_scaler_{10000}.pkl"
    output_dir = Path(image_chip.output_path)
    pickle_file_path = output_dir / pickle_file_name
    with open(pickle_file_path, "rb") as f:
        file_scaler = pickle.load(f)
    assert image_chip.standard_scaler == file_scaler


def test_pickle_custom_scaler(setup_output_dir):
    out_dir = setup_output_dir
    custom_pickle_dir = out_dir / "scaler_test.pkl"
    image_chip = ImageChip(
        input_image_path="tests/data/test_img.tif", output_path=out_dir
    )
    image_chip.set_scaler(write_path=custom_pickle_dir)
    with open(custom_pickle_dir, "rb") as f:
        file_scaler = pickle.load(f)
    assert image_chip.standard_scaler == file_scaler


def test_apply_normaliser_valid():
    array = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    normaliser_dict = {"min_val": [1], "max_val": [3]}
    expected_output = np.array([[[0.0, 0.5], [1.0, 1.0]]], dtype=np.float32)
    result = ImageChip.apply_normaliser(array, normaliser_dict)
    np.testing.assert_array_almost_equal(result, expected_output)


def test_apply_normaliser_invalid_band_length():
    array = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    normaliser_dict = {"min_val": [1, 4], "max_val": [2, 5]}
    with pytest.raises(
        ValueError,
        match="Array band dimension .* does not match length of normaliser clip values",
    ):
        ImageChip.apply_normaliser(array, normaliser_dict)


if __name__ == "__main__":
    pytest.main()
