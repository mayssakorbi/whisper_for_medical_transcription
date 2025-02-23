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




