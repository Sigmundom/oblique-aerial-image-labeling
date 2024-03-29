from setuptools import  find_packages, setup

setup(
    name='Skrafoto',
    version='0.1.0',
    py_modules=['skrafoto'],
    install_requires=[
        'Click',
        'Shapely',
        'rasterio',
        'numpy',
        'matplotlib',
        'pillow',
        'tqdm',
        'chardet',
        'pycocotools',
        'dbfread',
        'utm',
        'pyshp',
    ],
    packages=find_packages(where='src', ),
    package_dir = {"": "src"},
    entry_points={
        'console_scripts': [
            'semantic-segmentation = scripts:semantic_segmentation',
            'create-dataset = scripts:create_dataset',
            'prepare_analysis = scripts:prepare_analysis',
            'analyze_predictions = scripts:analyze_predictions',
            'evaluate = scripts:evaluate_analysis'
        ],
    },
)