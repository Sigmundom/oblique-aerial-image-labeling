from setuptools import setup

setup(
    name='Skråfoto',
    version='0.1.0',
    py_modules=['skrafoto'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'coco-reformat = coco_reformat:main',
            'semantic-segmentation = annotated_tiled_image:semantic_segmentation',
        ],
    },
)