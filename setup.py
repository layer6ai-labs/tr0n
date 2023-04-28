import os
from setuptools import setup
from pathlib import Path

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths

extra_files = package_files(os.path.join(str(Path(__file__).parent), "tr0n", "modules"))

setup(
    name='tr0n',
    packages=['tr0n'],
    version='1.0',
    description='Official code for TR0N',
    author='Layer 6 AI',
    package_data = {'tr0n': extra_files},
    install_requires = [
        'torch',
        'torchvision',
        'ftfy',
        'h5py',
        'regex',
        'tqdm',
        'Ninja',
        'transformers',
        'tensorboard',
        'pytorch-fid',
        'open-clip-torch',
        'pycocotools',
        'clip @ git+https://github.com/openai/CLIP.git'
    ]
)
