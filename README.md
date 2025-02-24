### 🏥 Fine-Tuning Speech-to-Text AI pour la Transcription Médicale

Ce projet a pour objectif d’adapter un modèle de reconnaissance vocale afin d’améliorer la transcription des consultations médicales et l’identification des termes médicaux spécifiques. 
L'entraînement et l'évaluation du modèle incluent des métriques telles que le Word Error Rate (WER), le Character Error Rate (CER) et la Medical Term Accuracy (MTA).

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

### 🏗️ Choix du Modèle

Nous avons opté pour Whisper Medium d'OpenAI en raison de son équilibre optimal entre performance et consommation de ressources. Ce modèle offre une excellente capacité de reconnaissance vocale automatique (ASR) tout en étant suffisamment léger pour un fine-tuning efficace sur des ressources GPU standards. Sa version Medium a été choisie plutôt que les variantes plus petites (moins précises) ou plus grandes (trop gourmandes en mémoire), car elle offre une bonne généralisation sur des données médicales tout en restant adaptable à des ajustements spécifiques.

Pour optimiser davantage l'entraînement, nous avons intégré QLoRA (Quantized Low-Rank Adaptation), une technique combinant la quantification en 4-bit et LoRA afin de réduire considérablement l'utilisation mémoire tout en préservant la qualité des performances. Cette approche permet de fine-tuner uniquement un sous-ensemble des poids du modèle, évitant ainsi de stocker l'intégralité du modèle en pleine précision sur le GPU. Grâce à QLoRA, nous avons pu réduire l'empreinte mémoire GPU, accélérer l'entraînement et rendre le fine-tuning plus accessible sur des infrastructures limitées, tout en conservant une précision élevée sur les termes médicaux.

## 🛠 1.Installation des dépendances
```python
!pip install torch
!pip install datasets --quiet
!pip install scikit-learn --quiet
!pip install pandas --quiet
!pip install matplotlib
!pip install scikit-learn
!pip install evaluate
!pip install torch torchaudio transformers datasets jiwer librosa soundfile
!pip install -q bitsandbytes datasets accelerate loralib transformers
!pip install -q git+https://github.com/huggingface/transformers.git@main 
!pip install ipywidgets
!pip install git+https://github.com/huggingface/transformers
!pip install librosa
!pip install jiwer
!pip install -U bitsandbytes
!pip install -U accelerate
!pip install -U transformers
!pip install -U torch
!pip install --upgrade bitsandbytes
```
## 🕵️‍♂️ 2. Exploration des Données

L'objectif de cette phase est d'analyser la structure et les caractéristiques des données disponibles afin d'identifier d'éventuels problèmes tels que le déséquilibre des classes, la présence de bruit dans les enregistrements ou encore des transcriptions erronées. Voici les étapes réalisées :

🚀 Avec cette analyse détaillée, nous garantissons une préparation optimale des données avant le fine-tuning !

## 📌 **Aperçu du Dataset**
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

## 📊 Vérification de la distribution des données: 
Nous avons vérifié la répartition des ensembles d'entraînement, de validation et de test pour nous assurer qu'ils sont bien équilibrés.

```python
import matplotlib.pyplot as plt

# Distribution du dataset
dataset_sizes = {
    "Train": 5900,
    "Validation": 680,
    "Test": 680
}
plt.figure(figsize=(8, 5))
plt.bar(dataset_sizes.keys(), dataset_sizes.values(), color=['blue', 'green', 'red'])
plt.xlabel("Ensemble de données")
plt.ylabel("Nombre d'échantillons")
plt.title("Distribution des données dans le dataset")
plt.show()
```
--> Observation : Le dataset est fortement déséquilibré avec un nombre significatif d’échantillons dans l’ensemble de test par rapport à l’entraînement. Cela pourrait impacter la performance du modèle. Il est nécessaire de rééquilibrer les classes.

### 🔎 Analyse des valeurs manquantes et dupliquées

Nous avons analysé les valeurs nulles et les doublons dans le dataset afin d’évaluer la qualité des données.

```python
df_summary = pd.DataFrame({
    "Nombre de valeurs nulles": df_all.isnull().sum(),
    "Pourcentage de valeurs manquantes": df_all.isna().sum(),
    "Nombre de doublons": df_all.duplicated().sum()
})
print(df_summary)
```
## 🎚️ Visualisation des taux d'échantillonnage

Nous avons analysé la distribution des taux d'échantillonnage pour identifier les écarts dans les données audio.
```python
# Extraire les taux d'échantillonnage
taux_echantillonnage = [row['path']['sampling_rate'] for _, row in df_all.iterrows()]

plt.figure(figsize=(8, 5))
plt.hist(taux_echantillonnage, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel("Taux d'échantillonnage (Hz)")
plt.ylabel("Nombre d'enregistrements")
plt.title("Distribution des taux d'échantillonnage des fichiers audio")
plt.grid(True)
plt.show()
```
Observation : Une normalisation à 16 kHz est nécessaire pour garantir une cohérence dans le traitement des données.

## 🔊 Diagramme des niveaux RMS (Root Mean Square)

L'analyse des niveaux RMS permet d'évaluer l'intensité sonore moyenne des fichiers audio.
```python
# Calcul des niveaux RMS
rms_levels = [np.sqrt(np.mean(row['path']['array']**2)) for _, row in df_all.iterrows()]

plt.figure(figsize=(8, 5))
plt.hist(rms_levels, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel("Énergie RMS")
plt.ylabel("Nombre de fichiers audio")
plt.title("Distribution de l'énergie RMS des fichiers audio")
plt.grid(True)
plt.show()
```
Observation : Certains fichiers présentent un niveau sonore très faible ou très élevé, ce qui pourrait nécessiter un filtrage supplémentaire.


## ⚠️ Analyse des transcriptions ambiguës

Nous avons recherché des caractères spéciaux inhabituels pouvant indiquer des erreurs dans les transcriptions.

```python
import re
pattern = r'[@#&^~+=<>$€¥¢£\…“”«»‘’¡¿]'
anomalous_transcriptions = df_all[
    df_all['sentence'].str.contains(pattern, regex=True, na=False)
]['sentence'].tolist()

print(f"Nombre de transcriptions contenant des caractères ambigus : {len(anomalous_transcriptions)}")
print("\n".join(anomalous_transcriptions[:10]))  # Afficher quelques exemples
```

Analyse du volume des enregistrements pour identifier d’éventuelles irrégularités.

## 🎵 Analyse des spectrogrammes

La visualisation des spectrogrammes permet d'évaluer la qualité des enregistrements audio.
```python
import librosa.display

def plot_spectrogram(audio_signal, sampling_rate):
    plt.figure(figsize=(10, 4))
    spectrogram = librosa.amplitude_to_db(librosa.stft(audio_signal), ref=np.max)
    librosa.display.specshow(spectrogram, sr=sampling_rate, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogramme")
    plt.show()

# Exemple d'affichage
audio_example = df_all.iloc[0]['path']['array']
sr_example = df_all.iloc[0]['path']['sampling_rate']
plot_spectrogram(audio_example, sr_example)
```
Observation : Certains fichiers montrent des niveaux de bruit élevés, ce qui confirme la nécessité d’un filtrage des fichiers les plus bruyants.

### 📏 Analyse des longueurs de transcription

Nous avons analysé la distribution des longueurs de transcription pour voir si elles sont cohérentes.
```python
# Calcul des longueurs des transcriptions
transcription_lengths = df_all['sentence'].apply(lambda x: len(str(x).split()))

# Affichage de l'histogramme
plt.figure(figsize=(10, 5))
plt.hist(transcription_lengths, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Nombre de mots par transcription")
plt.ylabel("Fréquence")
plt.title("Distribution des longueurs des transcriptions")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

Observation : Certaines transcriptions sont très courtes et pourraient ne pas contenir suffisamment d'informations utiles.


✅ Conclusion de l'Exploration des Données

- Le dataset est déséquilibré, nécessitant une meilleure gestion des échantillons.

- Les taux d’échantillonnage varient, donc une normalisation à 16 kHz est requise.

- Certains fichiers audio sont trop courts pour être utiles.

- Présence de bruit élevé dans certains fichiers audio, nécessitant un filtrage.

- Certains fichiers contiennent des caractères spéciaux, ce qui nécessite un nettoyage.

- Les transcriptions varient en longueur, il faut ajuster les hyperparamètres pour s'adapter à cette variabilité.

### 🔧 3.Prétraitement des Données

La phase de préparation des données est essentielle pour garantir un fine-tuning efficace du modèle. Cette étape inclut :

### 1️⃣ Filtrage des Audios Trop Bruyants

Avant l'entraînement, il est crucial de supprimer les fichiers audio contenant un bruit excessif. Nous avons mesuré le niveau de bruit à l'aide de plusieurs métriques acoustiques, notamment :

RMS Energy : mesure de l’intensité sonore globale.
```python
def detect_and_sort_noisy_files(df_all):
    noise_data = []

    for index, row in df_all.iterrows():
        y = row['path']['array']  
        energy = np.mean(np.abs(y))  
        noise_data.append((index, energy))  


    noise_df = pd.DataFrame(noise_data, columns=["index", "energy"])
    noise_df = noise_df.sort_values(by="energy", ascending=True)

    return noise_df 
sorted_noisy_files = detect_and_sort_noisy_files(df_all)

print("Top 5 fichiers les moins bruyants:")
print(sorted_noisy_files.head())
print("\nTop 5 fichiers les plus bruyants:")
print(sorted_noisy_files.tail())
most_noisy_index = sorted_noisy_files.iloc[-1]["index"]
most_noisy_audio = df_all.iloc[int(most_noisy_index)]['path']['array']
sr = df_all.iloc[int(most_noisy_index)]['path']['sampling_rate']
print("\nLecture du fichier audio le plus bruyant :")
ipd.Audio(most_noisy_audio, rate=sr)
```

Nous avons filtré les fichiers dépassant un seuil de bruit élevé, afin de conserver uniquement les enregistrements exploitables pour le fine-tuning.
Observation : Certains fichiers étaient extrêmement bruyants et nécessitaient d’être retirés du dataset pour éviter d’impacter la qualité de la transcription.

### 2️⃣ Conversion des Audios en 16 kHz

Pour assurer la compatibilité avec Whisper, nous avons converti tous les fichiers audio en mono 16 kHz
``` python

def convert_to_wav_16khz_mono(df):
    for index, row in df.iterrows(): 
        
        y = row['path']['array']
        sr = row['path']['sampling_rate']  

        if len(y.shape) > 1:
            y = librosa.to_mono(y)  

        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)  


        new_path = f"converted_audio_{index}.wav"
        sf.write(new_path, y, 16000)
        df.at[index, 'path']['array'] = y  
        df.at[index, 'path']['sampling_rate'] = 16000 
convert_to_wav_16khz_mono(df_cleaned)
```

Observation : Certains fichiers avaient des taux d’échantillonnage variés, d’où la nécessité de normaliser en 16 kHz pour garantir une cohérence des entrées du modèle.

### 3️⃣ Normalisation des Transcriptions

Les transcriptions brutes contiennent parfois des incohérences (majuscules, ponctuation superflue, caractères spéciaux). Une normalisation textuelle a donc été appliquée :

``` python
def normalize_transcription(text):
    """
    Nettoie et homogénéise une transcription médicale.
    :param text: Texte brut de la transcription
    :return: Texte nettoyé et normalisé
    """
    if not isinstance(text, str):
        return ""  # Gérer les valeurs non textuelles (NaN, None, etc.)

    text = text.lower()  # Convertir en minuscules
    text = re.sub(r'[^a-zA-Z,.!? ]', '', text)  # Supprimer chiffres et caractères spéciaux (sauf ponctuation)
    text = re.sub(r'\s+', ' ', text).strip()  # Supprimer les espaces en trop

    return text  # Retourner le texte nettoyé

# Assurer que df_cleaned est une copie indépendante
df_cleaned = df_cleaned.copy()

# Appliquer la normalisation de manière sûre
df_cleaned.loc[:, "sentence"] = df_cleaned["sentence"].apply(normalize_transcription)

# Vérifier quelques transcriptions normalisées
print(df_cleaned["sentence"].head())
```

### 4️⃣ Sélection des Attributs les Plus Pertinents avec Random Forest

Pour vérifier si le champ "prompt" et "speaker_id" influencent la transcription, nous avons utilisé RandomForestClassifier afin d’analyser l’importance de ces variables.
Exemple de code: 
``` python
# Encoder les phrases en valeurs numériques (ex: ID unique pour chaque phrase)
new_data['sentence_encoded']= new_data['sentence'].astype('category').cat.codes

# Définir les features et la cible
X = new_data[['speaker_id']]  # Feature testée
y = new_data['sentence_encoded'] # Phrase encodée

# Séparer en train / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner un Random Forest
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prédictions et évaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Précision du modèle basé uniquement sur `speaker_id` :", accuracy)

```
Observation : Le champ "prompt" et "speaker_id" n'ont pas d’impact sur les transcriptions.

5️⃣ Téléchargement du Modèle, Tokenizer, Feature Extractor et Processor

Avant le fine-tuning, nous avons téléchargé les composants nécessaires :
``` python
model_name_or_path = "openai/whisper-medium"  
task = "transcribe"  
language = "English"  
language_abbr = "en" 
from transformers import WhisperFeatureExtractor
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
from transformers import WhisperTokenizer
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)
```
Utilité :

- Feature Extractor : Convertit les signaux audio en log-Mel spectrogrammes.

- Tokenizer : Transforme les transcriptions textuelles en tokens pour le modèle.

- Processor : Rassemble ces deux outils pour assurer un prétraitement cohérent des données.


### 6️⃣ Préparation des Données pour le Modèle

Nous avons défini une fonction prepare_dataset pour traiter chaque échantillon :
``` python
def prepare_dataset(batch):
    audio = batch["path"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

medical_data = medical_data.map(prepare_dataset, remove_columns=medical_data.column_names["train"], num_proc=4)
```

✅ Conclusion de la Préparation des Données

- Les fichiers audio trop bruyants ont été supprimés.

- Tous les fichiers audio ont été convertis en 16 kHz mono.

- Les transcriptions ont été normalisées pour uniformiser les entrées.

- L'analyse des attributs a montré que "prompt" influence la transcription.

- Le dataset a été pré-traité et rendu compatible avec Whisper.
- 
### 🏁 4.Plan pour l’évaluation avant le fine-tuning
Avant d'entraîner le modèle Whisper sur notre dataset médical, il est essentiel d'évaluer ses performances initiales sur l'ensemble de validation. Cette étape nous permet d'avoir un point de comparaison après le fine-tuning et d’identifier les faiblesses du modèle sur notre domaine spécifique.

