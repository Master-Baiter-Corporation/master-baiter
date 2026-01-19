# Guide d'exécution des benchmarks

Ce guide explique comment exécuter les scripts de benchmark sur différentes plateformes.

## Table des matières
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Benchmark standard (CPU)](#benchmark-standard-cpu)
- [Benchmark Google Coral (Edge TPU)](#benchmark-google-coral-edge-tpu)
- [Interprétation des résultats](#interprétation-des-résultats)

---

## Prérequis

### Matériel requis
- **Pour benchmark standard** : MacBook M2, Raspberry Pi 4, ou autre ordinateur
- **Pour benchmark Coral** : Raspberry Pi 4 + Google Coral USB Accelerator

### Fichiers requis
```
votre_projet/
├── models/
│   ├── model_int8.tflite
│   ├── model_float16.tflite
│   ├── model_pruned_50.tflite
│   ├── model_pruned_70.tflite
│   ├── model_pruned_50_int8.tflite
│   ├── model_pruned_70_int8.tflite
│   ├── model_pruned_50_float16.tflite
│   └── (pour Coral) model_*_edgetpu.tflite
├── data/
│   └── sound_test/
│       └── a.wav (fichier audio de test de 30 secondes)
└── scripts/
    ├── benchmark_models.py
    └── test_audio_coral.py

```

---

## Installation

### 1. Installer Python et les dépendances

#### Mettre en place le fichier audio de test

Télécharger le fichier `a.wav` à ce lien [Google drive](https://drive.google.com/file/d/1tT084T0W7SYNBM1O1SYqbdSdIIRic2OI/view?usp=sharing) et le placer dans le dossier `data/sound_test/` (voir arbre au dessus).

#### Sur MacBook M2 ou ordinateur standard
```bash
# Créer un environnement virtuel (recommandé)
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install tensorflow numpy librosa
```

#### Sur Raspberry Pi 4 (sans Coral)
```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install tensorflow numpy librosa
```

#### Sur Raspberry Pi 4 (avec Coral)
```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer tflite-runtime (plus léger que TensorFlow complet)
pip install tflite-runtime numpy librosa

# Installer le support Google Coral
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
```

### 2. Vérifier l'installation

#### Test des dépendances standard
```bash
python3 -c "import tensorflow as tf; import numpy; import librosa; print('✅ OK')"
```

#### Test du Google Coral (si applicable)
```bash
# Vérifier que la clé Coral est détectée
lsusb | grep "Global Unichip"

# Tester pycoral
python3 -c "from pycoral.utils import edgetpu; print('✅ Coral OK')"
```

---

## Benchmark standard (CPU)

Ce script teste tous les modèles sur CPU (ou CPU avec accélération NEON sur Raspberry Pi).

### Exécution basique
```bash
cd scripts/
python3 benchmark_models.py
```

Par défaut, cela va :
- Tester le fichier audio `a.wav` dans `data/sound_test/`
- Évaluer tous les modèles disponibles
- Afficher les résultats pour chaque modèle
- Faire une pause entre chaque test

### Options avancées

#### Tester un autre fichier audio
```bash
python3 benchmark_models.py mon_audio.wav
```

#### Mode automatique (sans pause)
```bash
python3 benchmark_models.py --auto
```

#### Spécifier les vrais labels (si on choisi un autre fichier audio)
```bash
python3 benchmark_models.py mon_audio.wav --ground-truth avance recule droite gauche
```

---

## Benchmark Google Coral (Edge TPU)

Ce script teste uniquement les modèles compilés pour Edge TPU.

### Prérequis spécifiques

1. **Brancher le Google Coral USB Accelerator**

2. **Compiler les modèles pour Edge TPU**

Les modèles `.tflite` normaux ne fonctionnent pas sur le Coral. Vous devez les compiler. Pour ce benchmark, nous avons fournis les versions des modèles déjà compilé en edgetpu. Si besoin

#### Compiler sur colab
1. Allez sur https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Compile_for_edge_tpu.ipynb
2. Uploadez chaque fichier int8 dans le gestionaire de fichier colab
3. Executez les cellules en modifiant le nom du fichier `android.tflite` avec le nom de votre fichier
4. Téléchargez les versions `_edgetpu.tflite` générées
5. Placez-les dans le dossier `models/`

### Exécution
```bash
cd scripts/
python3 benchmark_models_coral.py
```

### Options

Les mêmes options que pour `benchmark_models.py` sont disponibles :
```bash
# Fichier audio spécifique
python3 benchmark_models_coral.py mon_audio.wav

# Mode automatique
python3 benchmark_models_coral.py --auto

# Ground truth personnalisé
python3 benchmark_models_coral.py mon_audio.wav --ground-truth avance recule droite

```

---

## Interprétation des résultats

### Métriques affichées

#### **Taille (KB)**
- Taille du fichier `.tflite` sur disque
- Plus petit = moins de mémoire requise

#### **Inference only (ms)**
- Temps moyen pour une seule prédiction
- Mesuré après 5 appels de warmup
- Plus petit = modèle plus rapide

#### **Détections**
- Liste des instructions détectées dans l'audio
- À comparer avec le ground truth

#### **Accuracy (Recall)**
- Proportion d'instructions réelles correctement détectées
- Formule : `détections correctes / total instructions réelles`
- **Important** : Mesure si le modèle manque des instructions
- Exemple : 11/12 = 91.67% → 1 instruction manquée

#### **Precision**
- Proportion de prédictions correctes parmi toutes les prédictions
- Formule : `détections correctes / total prédictions`
- **Important** : Mesure les fausses alarmes
- Exemple : 11/13 = 84.62% → 2 fausses détections

#### **Missing**
- Nombre d'instructions réelles non détectées (faux négatifs)
- Impact : Le robot ignore des commandes

#### **False (Extra)**
- Nombre de fausses détections (faux positifs)
- Impact : Le robot exécute des commandes erronées

#### **Temps d'exécution (s)**
- Temps total du pipeline complet :
  - Chargement audio
  - Découpage en segments
  - Transformation en spectrogrammes
  - Prédictions
  - Post-traitement
- Inclut les 30 secondes d'audio

### Comparaison entre plateformes

Lors de l'exécution sur différents environnements :

- **Accuracy et Precision** : Devraient être **identiques** (déterministe)
- **Temps d'exécution** : Varie selon la puissance de calcul
- **Inference only** : Meilleure métrique pour comparer les performances

---

## Répéter les benchmarks

Pour obtenir des résultats statistiquement significatifs, il est conseillé d'executer les benchmarks au moins 3 fois en notant les traces.

Ensuite, calculez les moyennes manuellement ou avec un script.
