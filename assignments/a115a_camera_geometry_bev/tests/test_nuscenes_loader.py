"""nuScenes loader: skips cleanly when the dataset/devkit is absent.

The loader is provided boilerplate, not a task. This test imports the module
(which must succeed even without nuscenes-devkit, thanks to lazy imports),
skips when NUSCENES_DATAROOT is unset or the devkit is missing, and otherwise
sanity-checks one sample's shapes.
"""

import os

import pytest


def test_loader_module_imports_without_devkit():
    # The module must be importable even when nuscenes-devkit is not installed:
    # the devkit import is lazy (inside NuScenesMini.__init__), not at module top.
    import nanovision.data.nuscenes_mini as m

    assert hasattr(m, "NuScenesMini")
    assert hasattr(m, "quaternion_wxyz_to_matrix")


def test_loader_sample_shapes():
    dataroot = os.environ.get("NUSCENES_DATAROOT")
    if not dataroot or not os.path.isdir(dataroot):
        pytest.skip("NUSCENES_DATAROOT unset or missing; see README step zero")
    try:
        import nuscenes  # noqa: F401
    except ImportError:
        pytest.skip("nuscenes-devkit not installed; run: pip install -e \".[av]\"")

    from nanovision.data.nuscenes_mini import NuScenesMini

    ds = NuScenesMini(dataroot=dataroot, image_size=(400, 224))
    assert len(ds) > 0
    sample = ds[0]
    assert sample["lidar"].shape[1] == 3
    for cam, img in sample["images"].items():
        assert img.shape[0] == 3
        assert img.shape[1] == 224 and img.shape[2] == 400
        assert sample["rig"].Ks[cam].shape == (3, 3)
        assert sample["rig"].extrinsics[cam].shape == (4, 4)
    assert sample["bev_grid"].nx == 200 and sample["bev_grid"].ny == 200
