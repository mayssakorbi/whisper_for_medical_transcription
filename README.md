### 🏥 Fine-Tuning Speech-to-Text AI pour la Transcription Médicale
Ce projet a pour objectif d’adapter un modèle de reconnaissance vocale afin d’améliorer la transcription des consultations médicales et l’identification des termes médicaux spécifiques. 
L'entraînement et l'évaluation du modèle incluent des métriques telles que le Word Error Rate (WER), le Character Error Rate (CER) et la Medical Term Accuracy (MTA).

### 🎯 Objectifs
✅ Améliorer la précision des transcriptions vocales médicales

✅ Réduire les erreurs sur les termes spécifiques au domaine médical

✅ Utiliser un modèle pré-entraîné et l’adapter aux données de consultation médicale

✅ Implémenter LoRA / QLoRA pour optimiser la mémoire et accélérer l’entraînement

### 📥 Données Utilisées
Ce projet utilise le dataset **Medical Speech, Transcription, and Intent**, initialement publié sur Kaggle et accessible via Hugging Face sous la référence :

```python
from datasets import load_dataset
ds = load_dataset("yashtiwari/PaulMooney-Medical-ASR-Data")
```
###  📌 Source des Données

Les données proviennent du projet Figure Eight, initialement publié sur Kaggle :

[🔗 Lien Kaggle : Medical Speech, Transcription, and Intent](https://www.kaggle.com/datasets/paultimothymooney/medical-speech-transcription-and-intent)

Ces enregistrements ont été collectés pour faciliter la formation d'agents conversationnels médicaux, en permettant une meilleure compréhension des symptômes exprimés par les patients.

###  📊 Caractéristiques du Dataset

| **Attributs**          | **Description** |
|------------------------|----------------|
| **Durée totale**       | 8,5 heures d’enregistrements audio |
| **Nombre d'énoncés**   | Plusieurs milliers de phrases sur des symptômes médicaux |
| **Format audio**       | WAV |
| **Fréquence d'échantillonnage** | Variable (min : 8 000 Hz, max : 192 000 Hz, moyenne : 49 093.32 Hz) |
| **Types de symptômes** | Maux de tête, douleurs articulaires, fièvre, fatigue, etc. |
| **Langue**             | Anglais |
| **Annotations**        | Transcriptions textuelles associées aux fichiers audio |


### 📁 Structure du Dataset

| **Partition**       | **Fichier**                           | **Taille**  |
|---------------------|--------------------------------------|------------|
| **Train Set**      | `patient_symptom_audio_train.zip`   | 160,2 MB   |
| **Validation Set** | `patient_symptom_audio_validate.zip` | 137,7 MB   |
| **Test Set**       | `patient_symptom_audio_test.zip`    | 2,3 GB     |
| **Métadonnées**    | `recordings-overview.csv`           | 1,7 MB     |

| **Nom de colonne**   | **Type**        | **Description** |
|----------------------|----------------|----------------|
| `id`                | `string`        | Identifiant unique de l'énoncé |
| `sentence`          | `string`        | Transcription textuelle associée à l'audio |
| `prompt`            | `string`        | Types de symptômes |
| `speaker_id`        | `int64`         | Identifiant unique du locuteur |
| `path`              | `dict` (Audio)  | Dictionnaire contenant :  **`sampling_rate`** (`int`): Fréquence d'échantillonnage de l'audio  **`array`** (`numpy.ndarray`): Signal audio sous forme de tableau numérique   **`path`** (`string`): Chemin du fichier audio 


### 🕵️‍♂️ Exploration des Données

📌 Pourquoi cette exploration est essentielle ?

✅ Optimise l'entraînement du modèle en détectant les erreurs et incohérences en amont.

✅ Améliore la qualité des transcriptions en identifiant les enregistrements problématiques.

✅ Assure une meilleure performance du modèle de reconnaissance vocale sur des données réalistes.


🚀 Avec cette analyse détaillée, nous garantissons une préparation optimale des données avant le fine-tuning !

📌 **Aperçu du Dataset**
Chargement et affichage de la premiére lignes du dataset :

```python
from datasets import load_dataset

# Charger le dataset
ds = load_dataset("yashtiwari/PaulMooney-Medical-ASR-Data")

# Afficher quelques échantillons
print(ds["train"][0])
```
Exemple de sortie
```python
{'id': '1249120_44160489_107692984',
'sentence': 'My cough is very heavy and I have mucus.',
'prompt': 'Cough',
'speaker_id': 44160489,
'path': {'path': '1249120_44160489_107692984.wav',
'array': array([-0.01638794, -0.01553345, -0.01733398, ...,  0.02438354,
        0.02389526,  0.02334595]),
'sampling_rate': 44100}}
```
 📏 1️⃣ Distribution des Durées Audio
 
Analyse de la répartition des durées des enregistrements pour détecter d’éventuelles anomalies (audios trop courts ou trop longs).

🎚 2️⃣ Distribution des Taux d'Échantillonnage

Vérification de la variabilité des taux d’échantillonnage (min, max, moyenne).

🔊 3️⃣ Mesure du Loudness (Niveau Sonore)

Analyse du volume des enregistrements pour identifier d’éventuelles irrégularités.

🎼 4️⃣ Visualisation des Spectrogrammes pour la Qualité Audio

Inspection des spectrogrammes pour détecter les enregistrements bruyants, les distorsions ou les pertes de signal.

📜 5️⃣ Distribution des Longueurs des Transcriptions

Analyse de la longueur des phrases pour s’assurer qu’elles ne sont ni trop courtes ni trop longues.

🔍 6️⃣ Détection des Transcriptions Ambiguës

Identification des transcriptions contenant des erreurs, des répétitions ou des formulations imprécises.

⚠️ 7️⃣ Vérification des Valeurs Manquantes, Duplicates et Données de Faible Qualité

Détection des enregistrements sans transcription ou avec des transcriptions vides.

🔤 8️⃣ Vérification de la Diversité Lexicale du Dataset

Analyse de la richesse du vocabulaire utilisé dans les transcriptions.

🎙 9️⃣ Calcul du Nombre d'Audios Considérés comme Bruyants

Évaluation du niveau de bruit de fond dans chaque enregistrement.

### 🔄 Prétraitement des Données

Après l’exploration du dataset, il est crucial de normaliser et nettoyer les données pour assurer un entraînement optimal du modèle. 
Cette phase vise à filtrer les enregistrements de mauvaise qualité, standardiser le format des fichiers audio et préparer les transcriptions pour le fine-tuning.

📌 Objectifs du prétraitement :

✅ Filtrer les fichiers audio bruyants en définissant un seuil de bruit pour exclure les enregistrements trop dégradés.

✅ Convertir tous les fichiers audio en un format WAV, 16 kHz, mono pour assurer une cohérence avec le modèle.

✅ Nettoyer les transcriptions en supprimant les caractères ambigus (*, /, %, etc.).

✅ Uploader le tokenizer et le feature extractor de la bibliothèque Transformers pour appliquer des transformations normalisées sur les données audio et textuelles.

🔊 1️⃣ Détection et Suppression des Audios Trop Bruyants

Certains enregistrements contiennent un niveau de bruit trop élevé, pouvant nuire à l'entraînement.

Nous définissons un seuil de bruit basé sur le rapport signal/bruit (SNR) pour exclure ces fichiers.

Seuls les fichiers avec un niveau de bruit acceptable seront conservés.

🎙 2️⃣ Conversion et Normalisation des Audios

Les fichiers du dataset présentent des fréquences d’échantillonnage variées (min : 8 000 Hz, max : 192 000 Hz).

Pour garantir une compatibilité avec les modèles pré-entraînés, tous les fichiers sont :

🔹 Convertis en format WAV

🔹 Rééchantillonnés à 16 kHz

🔹 Passés en mono pour homogénéiser les entrées audio

📝 3️⃣ Nettoyage et Prétraitement des Transcriptions

Les transcriptions brutes peuvent contenir des caractères spéciaux inutiles (*, /, %...)

Pour assurer une cohérence linguistique, nous appliquons :

🔹 Suppression des caractères ambigus et symboles non pertinents

🔹 Normalisation des espaces et ponctuation

🔹 Conversion en minuscules pour uniformiser le texte


📚 4️⃣ Chargement du Processor (Tokeniseur + Feature Extractor)

Le Processor est une classe combinée qui regroupe à la fois :

1- Le Tokenizer : Responsable de la transformation des textes en tokens utilisables par le modèle.

2- Le Feature Extractor : Utilisé pour le prétraitement des fichiers audio afin de les normaliser et les rendre exploitables par le modèle.









