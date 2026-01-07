from setuptools import setup
from glob import glob
import os

package_name = 'speech_recognition'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.tflite')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Master Baiter Team',
    maintainer_email='dev@masterbaiter.com',
    description='Speech recognition node using TFLite for Master Baiter robot',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speech_recognition_node = speech_recognition.speech_recognition_node:main',
        ],
    },
)
