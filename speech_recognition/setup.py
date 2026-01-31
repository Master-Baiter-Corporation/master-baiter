from setuptools import setup
from glob import glob
import os

package_name = 'speech_recognition'

def get_vosk_model_files():
    """
    Récupère tous les fichiers du modèle Vosk (en excluant les dossiers)
    et retourne une liste de tuples (destination, [fichiers])
    """
    model_base = 'models/vosk-model-small-fr-0.22'
    data_files = []
    
    for root, dirs, files in os.walk(model_base):
        if files:  # Seulement si des fichiers existent dans ce dossier
            # Calculer le chemin de destination relatif
            rel_path = os.path.relpath(root, 'models')
            dest = os.path.join('share', package_name, 'models', rel_path)
            
            # Lister tous les fichiers de ce dossier
            file_paths = [os.path.join(root, f) for f in files]
            data_files.append((dest, file_paths))
    
    return data_files

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
        *get_vosk_model_files(),
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