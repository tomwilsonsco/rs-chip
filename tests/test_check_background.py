import pytest
from pathlib import Path
import tempfile
from rschip import ImageChip
from rschip import SegmentationMask
from rschip import CheckBackgroundOnly
import pandas as pd


@pytest.fixture(scope="function")
def setup_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def chip_image_run(
    output_path,
    input_image_path,
    output_name=None,
    pixel_dimensions=128,
    offset=64,
    use_multiprocessing=True,
    max_batch_size=10,
):
    image_chip = ImageChip(
        input_image_path=input_image_path,
        output_path=output_path,
        output_name=output_name,
        pixel_dimensions=pixel_dimensions,
        offset=offset,
        use_multiprocessing=use_multiprocessing,
        max_batch_size=max_batch_size,
    )
    image_chip.chip_image()


def tif_files_to_list(out_dir):
    return list(Path(out_dir).glob("*.tif"))


def test_background_check(setup_output_dir):
    out_dir = setup_output_dir
    out_mask = out_dir / "output_mask.tif"
    out_mask_chips = out_dir / "mask_chips"
    out_img_chips = out_dir / "img_chips"

    # Create directories for chips
    out_mask_chips.mkdir()
    out_img_chips.mkdir()

    mask_creator = SegmentationMask(
        "tests/data/test_img.tif", "tests/data/test_features.gpkg", out_mask
    )
    mask_creator.create_mask()

    chip_image_run(
        output_path=out_img_chips,
        input_image_path="tests/data/test_img.tif",
    )
    chip_image_run(
        output_path=out_mask_chips,
        input_image_path=out_mask,
        output_name="test_img",
    )

    tif_img_files_init = len(list(out_img_chips.glob("*.tif")))
    tif_mask_files_init = len(list(out_mask_chips.glob("*.tif")))

    checker = CheckBackgroundOnly(background_val=0, non_background_min=100)

    df = checker.check_background_chips(str(out_mask_chips), str(out_img_chips))

    tif_img_files_final = len(list(out_img_chips.glob("*.tif")))
    tif_mask_files_final = len(list(out_mask_chips.glob("*.tif")))

    # Check that no files were removed
    assert tif_img_files_final == tif_img_files_init, "Image files were removed"
    assert tif_mask_files_final == tif_mask_files_init, "Mask files were removed"

    # Check that CSV was created
    csv_path = out_mask_chips / "background_only_check.csv"
    assert csv_path.exists(), "Audit CSV file was not created."

    # Check DataFrame and CSV content
    assert len(df) == tif_mask_files_init
    assert "is_background_only" in df.columns
    assert df[
        "is_background_only"
    ].any(), "Expected some background-only chips to be True."
    assert not df[
        "is_background_only"
    ].all(), "Expected some non-background chips to be False."

    csv_df = pd.read_csv(csv_path)
    assert len(csv_df) == len(df)


def test_non_background_min_check(setup_output_dir):
    out_dir = setup_output_dir
    out_mask = out_dir / "output_mask.tif"
    out_mask_chips = out_dir / "mask_chips"
    out_img_chips = out_dir / "img_chips"

    mask_creator = SegmentationMask(
        "tests/data/test_img.tif", "tests/data/test_features.gpkg", out_mask
    )
    mask_creator.create_mask()

    chip_image_run(
        output_path=out_img_chips,
        input_image_path="tests/data/test_img.tif",
    )
    chip_image_run(
        output_path=out_mask_chips,
        input_image_path=out_mask,
        output_name="test_img",
    )
    # Audit with a low threshold
    checker1 = CheckBackgroundOnly(background_val=0, non_background_min=1)
    df1 = checker1.check_background_chips(str(out_mask_chips), str(out_img_chips))
    background_only_count1 = df1["is_background_only"].sum()

    # Audit with a high threshold
    checker2 = CheckBackgroundOnly(background_val=0, non_background_min=10000)
    df2 = checker2.check_background_chips(str(out_mask_chips), str(out_img_chips))
    background_only_count2 = df2["is_background_only"].sum()

    assert (
        background_only_count2 >= background_only_count1
    ), "Higher non_background_min threshold should result in more background-only chips."


def test_multiprocessing_consistency(setup_output_dir):
    """
    Ensure that running the check with and without multiprocessing yields the same results.
    """
    out_dir = setup_output_dir
    out_mask = out_dir / "output_mask.tif"
    out_mask_chips = out_dir / "mask_chips"
    out_img_chips = out_dir / "img_chips"

    out_mask_chips.mkdir()
    out_img_chips.mkdir()

    mask_creator = SegmentationMask(
        "tests/data/test_img.tif", "tests/data/test_features.gpkg", out_mask
    )
    mask_creator.create_mask()

    chip_image_run(
        output_path=out_img_chips,
        input_image_path="tests/data/test_img.tif",
    )
    chip_image_run(
        output_path=out_mask_chips,
        input_image_path=out_mask,
        output_name="test_img",
    )

    # run with multiprocessing
    checker_mp = CheckBackgroundOnly(use_multiprocessing=True)
    df_mp = checker_mp.check_background_chips(str(out_mask_chips), str(out_img_chips))

    # run sequentially
    checker_sp = CheckBackgroundOnly(use_multiprocessing=False)
    df_sp = checker_sp.check_background_chips(str(out_mask_chips), str(out_img_chips))

    # compare results
    pd.testing.assert_frame_equal(df_mp, df_sp)
