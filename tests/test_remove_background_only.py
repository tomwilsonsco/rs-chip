import pytest
from pathlib import Path
import tempfile
from rschip import ImageChip
from rschip import SegmentationMask
from rschip import RemoveBackgroundOnly


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


# repeat the same but for tif file chips not npz
def test_tif_remove(setup_output_dir):
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

    tif_img_files_init = len(list(out_img_chips.glob("*.tif")))
    tif_mask_files_init = len(list(out_mask_chips.glob("*.tif")))

    remover = RemoveBackgroundOnly(background_val=0, non_background_min=100)

    remover.remove_background_only_files(out_mask_chips, out_img_chips)

    tif_img_files_final = len(list(out_img_chips.glob("*.tif")))
    tif_mask_files_final = len(list(out_mask_chips.glob("*.tif")))

    assert tif_img_files_final < tif_img_files_init, "No img files were removed"

    assert tif_mask_files_final < tif_mask_files_init, "No chip files were removed"

    assert (
        tif_mask_files_final == tif_img_files_final
    ), "Remaining chips and image file number differs"


# specifying a large and small background threshold and compare
def test_non_background_min(setup_output_dir):
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
    remover = RemoveBackgroundOnly(background_val=0, non_background_min=1)
    remover.remove_background_only_files(out_mask_chips, out_img_chips)

    tif_img_files_final1 = len(list(out_img_chips.glob("*.tif")))
    tif_mask_files_final1 = len(list(out_mask_chips.glob("*.tif")))

    remover = RemoveBackgroundOnly(background_val=0, non_background_min=10000)
    remover.remove_background_only_files(out_mask_chips, out_img_chips)

    tif_img_files_final2 = len(list(out_img_chips.glob("*.tif")))
    tif_mask_files_final2 = len(list(out_mask_chips.glob("*.tif")))

    assert (
        tif_img_files_final2 < tif_img_files_final1
    ), "Non background threshold on images failed"

    assert (
        tif_mask_files_final2 < tif_mask_files_final1
    ), "Non background threshold on masks failed"
