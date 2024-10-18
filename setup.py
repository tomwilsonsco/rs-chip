from setuptools import setup, find_packages

setup(
    name="rschip",
    version="0.1",
    packages=find_packages(),
    install_requires=["rasterio", "numpy", "geopandas", "shapely"],
    tests_require=[
        "pytest",
    ],
    test_suite="tests",  # This tells setuptools to look in the 'tests' directory for tests
)
