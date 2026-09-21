from importlib.metadata import version

import maritime_cbm


def test_package_version() -> None:
    assert maritime_cbm.__version__ == version("maritime-cbm")
