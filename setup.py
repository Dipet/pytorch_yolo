import io
import os
import re
from setuptools import setup
from pkg_resources import DistributionNotFound, get_distribution


INSTALL_REQUIRES = [
    "torch>=1.2.0",
    "efficientnet-pytorch>=0.2.0",
    "matplotlib>=2.2.3",
    "numpy>=1.16.3",
    "pycocotools>=2.1",
    "tensorboardX>=1.9",
    "torchvision>=0.3.0",
    "tqdm",
]
CHOOSE_INSTALL_REQUIRES = [["opencv-python", "opencv-python-headless"]]


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "pytorch_yolo", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def choose_requirement(main, secondary):
    """If some version version of main requirement installed, return main,
    else return secondary.

    """
    try:
        name = re.split(r"[!<>=]", main)[0]
        get_distribution(name)
    except DistributionNotFound:
        return secondary

    return str(main)


def get_install_requirements(install_requires, choose_install_requires):
    for main, secondary in choose_install_requires:
        install_requires.append(choose_requirement(main, secondary))

    return install_requires


setup(
    name="pytorch_yolo",
    version=get_version(),
    description="YOLO models implementation",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Mikhail Druzhinin",
    license="MIT",
    install_requires=get_install_requirements(INSTALL_REQUIRES, CHOOSE_INSTALL_REQUIRES),
)
