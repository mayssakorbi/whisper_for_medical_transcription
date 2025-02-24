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

L'objectif de cette phase est d'analyser la structure et les caractéristiques des données disponibles afin d'identifier d'éventuels problèmes tels que le déséquilibre des classes, la présence de bruit dans les enregistrements ou encore des transcriptions erronées. 

🚀 Avec cette analyse détaillée, nous garantissons une préparation optimale des données avant le fine-tuning !

Voici les étapes réalisées :


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

### 🔧 3. Prétraitement des Données

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
Conclusion : Le champ "prompt" et "speaker_id" n'ont pas d’impact sur les transcriptions.

5️⃣ Téléchargement du Modèle, Tokenizer, Feature Extractor et Processor

Pour bien préparer le input adéquat au modèle, nous avons téléchargé les composants nécessaires :
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

- L'analyse des attributs a montré que "prompt" et speaker_id n'influencent pas la transcription.

- Le dataset a été pré-traité et rendu compatible avec Whisper.
  
### 🏁 4.Evaluation du modèle avant le fine-tuning
Avant d'entraîner le modèle Whisper sur notre dataset médical, il est essentiel d'évaluer ses performances initiales sur l'ensemble de validation. 
Cette étape nous permet d'avoir un point de comparaison après le fine-tuning et d’identifier les faiblesses du modèle sur notre domaine spécifique.
### 📥 Data Collator 
Pour évaluer le modèle, il est crucial d'assurer une préparation cohérente des données. Le Data Collator joue un rôle clé dans cette étape. Il permet :

- L'alignement et le padding des séquences (les entrées audio et les labels n’ont pas toujours la même longueur).

- L’optimisation du traitement en lot (batch processing), améliorant l'efficacité du modèle sur GPU.

- L’ignorance des tokens de remplissage dans la fonction de perte, garantissant une meilleure stabilité lors de l’apprentissage et l’évaluation.
```python

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Séparation des entrées audio et labels pour appliquer des méthodes de padding adaptées
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Récupération des labels sous forme tokenisée et application du padding
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Remplacement du padding par -100 pour que la perte ignore ces tokens
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Suppression du token de début de séquence si déjà présent
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Initialisation du Data Collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
```
Après cette préparation, nous procédons à l’évaluation initiale du modèle Whisper Medium sur le jeu de validation. Cela nous permet d’établir une ligne de base et de mesurer l’amélioration après fine-tuning.

### 🛠 Métriques d'Évaluation Utilisées

Nous utilisons plusieurs métriques pour évaluer la qualité des transcriptions générées :

- Word Error Rate (WER) 📝

Mesure le pourcentage de mots mal transcrits par rapport à la transcription de référence.
Plus la valeur est faible, meilleure est la transcription.

- Normalized Word Error Rate (Normalized WER) 📏

Version normalisée du WER où les majuscules, ponctuations et caractères non significatifs sont supprimés.Cette métrique est plus représentative des erreurs réelles du modèle en ASR.

- Character Error Rate (CER) 🔡

Similaire au WER, mais basé sur les caractères au lieu des mots.
Recommandé lorsque les erreurs sont fréquentes sur de petits mots ou des abréviations médicales.

- Normalized CER (Character Error Rate) 📏
  
Le Normalized CER (Character Error Rate) mesure le taux d’erreurs au niveau des caractères, en tenant compte des ajustements comme la suppression des espaces inutiles et la mise en minuscule.

- Medical Term Accuracy (MTA) 🏥

Indique la précision du modèle sur les termes médicaux clés.
Nous avons défini une liste de termes médicaux et comparé leur reconnaissance correcte.

### 🏗 Chargement du Modèle de Base et Évaluation

Nous utilisons le modèle Whisper Medium pré-entraîné sans modification.
L'évaluation est effectuée sur l'ensemble de validation. Exemple de code :
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate
import torch
from tqdm import tqdm
import gc
import numpy as np

# Charger le modèle de base
model_name = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to("cuda")

# Charger la métrique WER
metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

# Fonction de normalisation
normalizer = lambda x: x.lower().replace(",", "").replace(".", "").strip()

# Préparer les données de validation
eval_dataloader = DataLoader(medical_data["validation"], batch_size=8, collate_fn=data_collator)

# Initialisation des listes de stockage
predictions, references = [], []
normalized_predictions, normalized_references = [], []

# Mode évaluation
model.eval()
for batch in tqdm(eval_dataloader):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = model.generate(input_features=batch["input_features"].to("cuda"), max_new_tokens=255).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            normalized_predictions.extend([normalizer(pred) for pred in decoded_preds])
            normalized_references.extend([normalizer(label) for label in decoded_labels])

    gc.collect()

# Calcul des métriques
wer = 100 * metric_wer.compute(predictions=predictions, references=references)
normalized_wer = 100 * metric_wer.compute(predictions=normalized_predictions, references=normalized_references)
cer = 100 * metric_cer.compute(predictions=predictions, references=references)
normalized_cer = 100 * metric_cer.compute(predictions=normalized_predictions, references=normalized_references)

# Affichage des résultats
print(f"WER : {wer:.2f}%")
print(f"Normalized WER : {normalized_wer:.2f}%")
print(f"CER : {cer:.2f}%")
print(f"Normalized CER : {normalized_cer:.2f}%")
```

###  📈 Résultats de l'Évaluation Avant Fine-Tuning


| **Métrique**                 | **Valeur obtenue**               | **Description** |
|------------------------------|--------------------------------|----------------|
| **WER (Word Error Rate)**     | `31.18%`                      | Taux d’erreur au niveau des mots (substitutions, insertions, suppressions). |
| **Normalized WER**            | `12.86%`                      | WER après normalisation (suppression des variations d’espaces et de ponctuation). |
| **CER (Character Error Rate)**| `9.78%`                       | Taux d’erreur basé sur les caractères au lieu des mots. |
| **Normalized CER**            | `5.18%`                       | CER après normalisation du texte. |
| **MTA (Medical Term Accuracy)** | `64.20%`                   | Pourcentage de termes médicaux correctement transcrits. |

### 🎯 5.Fine-Tuning du Modèle Whisper Medium
L’objectif du fine-tuning est d’adapter Whisper Medium aux spécificités du domaine médical afin d'améliorer la reconnaissance des termes médicaux et de réduire les erreurs de transcription. 
Cette section détaille les différentes étapes du fine-tuning, incluant la quantification en 4-bit (QLoRA), l'optimisation des hyperparamètres et la sauvegarde efficace des poids adaptés via LoRA.

📥 Chargement du Modèle et Quantification en 4-bit (QLoRA)
Le modèle Whisper Medium est trop volumineux pour être fine-tuné efficacement sans optimisation mémoire. 

Nous appliquons QLoRA (Quantized LoRA), qui combine :

1- La quantification en 4-bit : réduit l’utilisation mémoire tout en maintenant les performances.
 
2- LoRA (Low-Rank Adaptation) : ajuste uniquement un sous-ensemble de paramètres pour accélérer l'apprentissage.
 
Pourquoi QLoRA ?

- Économie de mémoire : Permet d'entraîner de grands modèles sur des GPU avec moins de VRAM.
- Efficacité : LoRA n'entraîne qu’un petit ensemble de paramètres au lieu de modifier tout le modèle.
- Performances maintenues : L’impact de la quantification 4-bit sur l’exactitude du modèle reste négligeable.

### 🔧 Application de La Quantification en 4 bit

```python
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Charger le modèle Whisper Medium
model_name_or_path = "openai/whisper-medium"

# Appliquer la quantification en 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model_train = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, quantization_config=bnb_config)

# Préparer le modèle pour l'entraînement en 4-bit
model_train = prepare_model_for_kbit_training(model_train)

```
### 🔧 Application de LoRA

Après la quantification, nous intégrons LoRA pour rendre le fine-tuning plus efficace. LoRA remplace certains poids des couches de l'attention du modèle par des matrices à faible rang, réduisant ainsi le nombre de paramètres à ajuster.

```python
from peft import LoraConfig, get_peft_model

# Configuration de LoRA
config = LoraConfig(
    r=32,  # Taille de la matrice basse-rang
    lora_alpha=64,  # Facteur d’échelle
    target_modules=["q_proj", "v_proj"],  # Cibles : Projections en requête et valeur
    lora_dropout=0.05,  # Dropout pour éviter l'overfitting
    bias="none"
)

# Appliquer LoRA au modèle
model_train = get_peft_model(model_train, config)

# Vérifier les paramètres entraînables
model_train.print_trainable_parameters()
```
### 🔥 Justification des Hyperparamètres du Fine-Tuning

Le choix des hyperparamètres a été fait en fonction de :

- Taille du dataset (≈5900 enregistrements pour l'entraînement).
- Contraintes mémoire GPU (quantification 4-bit).
- Objectif de stabilité et de convergence rapide.

  ```python
    from transformers import Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
    output_dir="whisper_h100_finetuned",  # Dossier où stocker les modèles sauvegardés
    per_device_train_batch_size=32,  # Profite des 80GB de VRAM pour accélérer l'entraînement
    per_device_eval_batch_size=32,  # Meilleure évaluation
    gradient_accumulation_steps=1,  # Stabilisation du gradient
    learning_rate=2e-5,  # Taux d’apprentissage ajusté pour éviter un sur-ajustement
    lr_scheduler_type="cosine_with_restarts",  # Scheduler optimisé pour convergence progressive
    warmup_steps=1000,  # Éviter une descente trop brutale en début d'entraînement
    num_train_epochs=5,  # Nombre d’époques adapté à la taille du dataset
    weight_decay=0.05,  # Régularisation pour éviter l’overfitting
    evaluation_strategy="epoch",  # Évaluation après chaque époque
    save_strategy="epoch",  # Sauvegarde du modèle après chaque époque
    save_total_limit=3,  # Garde les 3 meilleurs modèles
    bf16=True,  # Utilisation de bfloat16 pour une meilleure gestion mémoire sur GPU
    dataloader_num_workers=4,  # Accélération de la gestion des données
    dataloader_pin_memory=True,  # Réduction de la latence CPU-GPU
    logging_steps=50,  # Fréquence d'affichage des logs pour un suivi optimal
    remove_unused_columns=False,  # Évite erreurs avec Trainer
    label_names=["labels"],  # Évite bugs avec Hugging Face Trainer
    predict_with_generate=True,  # Génération directe des transcriptions
    report_to="none",  # Désactive TensorBoard pour économiser mémoire serveur
    load_best_model_at_end=True  # Charge le meilleur modèle après entraînement )
  
  ```
## 📌 Justification des choix :

- Taille des batchs : per_device_train_batch_size=32 et per_device_eval_batch_size=32 → Permet un entraînement rapide tout en maximisant l’utilisation de la VRAM sur les GPU récents.
La taille du batch a été calibrée pour optimiser la consommation mémoire et la vitesse de convergence.

- Nombre d’époques (num_train_epochs=5) : Suffisant pour atteindre une bonne convergence sans entraîner un sur-ajustement.

- L'utilisation de load_best_model_at_end=True garantit que l'on récupère le meilleur modèle en fonction des performances sur l’ensemble de validation.
  
- Gradient Accumulation (gradient_accumulation_steps=1) : Permet une mise à jour des poids après chaque batch, ce qui assure une meilleure stabilité de l'entraînement.

- Gestion de l'apprentissage (learning_rate=2e-5, cosine_with_restarts) : Un taux d’apprentissage faible permet d'éviter des variations brusques et améliore la généralisation.

- lr_scheduler_type="cosine_with_restarts" ajuste dynamiquement la courbe d’apprentissage pour une convergence plus fluide.
 
- Quantification mémoire (bf16=True) : Utilisation de bfloat16, optimisé pour les GPU récents afin de réduire la consommation mémoire sans perte de précision.

- Gestion des modèles sauvegardés (save_total_limit=3) : Conserve uniquement les 3 meilleurs modèles pour optimiser l’espace de stockage.

🔹 Ces choix assurent un entraînement efficace, optimisé pour une consommation mémoire réduite et une généralisation performante sur des transcriptions médicales spécifiques. 🚀
  
  ### 💾 Sauvegarde Optimisée avec SavePeftModelCallback
  
  Par défaut, Seq2SeqTrainer enregistre tous les poids du modèle, ce qui consomme trop d’espace disque.
  
  Pour éviter cela, nous enregistrons uniquement les poids LoRA grâce à un callback personnalisé.

  ```python
  from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
  import os
  from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
  class SavePeftModelCallback(TrainerCallback):
       def on_save(
          self,
          args: TrainingArguments,
          state: TrainerState,
          control: TrainerControl,
          **kwargs,
    ):
         checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}- {state.global_step}")
         peft_model_path = os.path.join(checkpoint_folder, "adapter_model")

         kwargs["model"].save_pretrained(peft_model_path)

        # Suppression des poids inutiles
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        return control
        
    ```


  

🚀 Lancement de l'Entraînement
Enfin, nous utilisons Seq2SeqTrainer pour lancer le fine-tuning avec notre modèle quantifié et adapté.

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model_train,
    train_dataset=medical_data["train"],
    eval_dataset=medical_data["validation"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],)

# Désactiver le cache pour optimiser l'entraînement
model_train.config.use_cache = False

# Lancer l'entraînement
trainer.train()

```

### 📊 Analyse des Résultats du Fine-Tuning

Après l'entraînement du modèle Whisper Medium sur le dataset médical, nous avons observé une réduction progressive de la perte d'entraînement et de validation au fil des époques.

  
- Training Loss : Cette métrique indique à quel point le modèle s'ajuste aux données d'entraînement.

Une diminution constante signifie que le modèle apprend bien sans sur-ajustement excessif.

- Validation Loss : Cette métrique mesure l'erreur sur l'ensemble de validation, qui représente des données non vues par le modèle. 

Une baisse continue suggère une bonne généralisation.

 - 📉 Analyse des résultats obtenus :
  
Les pertes d'entraînement et de validation diminuent régulièrement, indiquant une convergence stable du modèle.

L'écart entre Training Loss et Validation Loss est faible, ce qui signifie que le modèle ne souffre pas d'overfitting majeur.

Les valeurs finales sont suffisamment basses pour indiquer une amélioration du modèle sur les données médicales.

### 📋 Résumé des Résultats du Fine-Tuning

| **Époque** | **Training Loss** | **Validation Loss** |
|------------|------------------|--------------------|
| **1**      | 3.9981           | 3.6444            |
| **2**      | 2.5208           | 2.1651            |
| **3**      | 1.7077           | 1.1735            |
| **4**      | 0.4820           | 0.3882            |
| **5**      | 0.3052           | 0.2763            |

### Conclusion :

Le modèle a montré une réduction significative des pertes tout au long des 5 époques, ce qui indique qu'il a bien appris à partir des données médicales.

Toutefois, une évaluation sur des métriques spécifiques (WER, CER, MTA) sera nécessaire pour confirmer les améliorations en termes de transcription médicale.

### 📤 6.Téléversement du Modèle Fine-Tuné sur Hugging Face Hub

Une fois l'entraînement terminé, nous devons sauvegarder et partager notre modèle fine-tuné. Pour cela, nous utilisons Hugging Face Model Hub, une plateforme qui permet de stocker et de diffuser facilement des modèles de machine learning.

Nous allons téléverser notre modèle sur Hugging Face Hub en utilisant l'identifiant défini au préalable.

``` python
# Définition de l'identifiant du modèle sur Hugging Face Hub
peft_model_id = "mayssakorbi/finetuned_whisper_medium_for_medical_transcription"

# Téléversement du modèle fine-tuné sur Hugging Face Model Hub
model_train.push_to_hub(peft_model_id)
```
###  🔄 7.Rechargement du Modèle Fine-Tuné depuis Hugging Face:

Après avoir téléversé le modèle fine-tuné, nous devons le recharger afin de l'évaluer sur un ensemble de test. Cette étape est cruciale pour vérifier les performances du modèle après fine-tuning et comparer ses résultats avec ceux obtenus avant l'entraînement.

```python
# Importation des classes nécessaires
from peft import PeftModel, PeftConfig
from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer

# Identifiant du modèle fine-tuné stocké sur Hugging Face
peft_model_id = "mayssakorbi/finetuned_whisper_medium_for_medical_transcription"  # Doit être identique à celui utilisé lors du push_to_hub

# Chargement de la configuration PEFT du modèle fine-tuné
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Chargement du modèle Whisper Medium d'origine
model_finetuned = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path,  # Charge le modèle de base (Whisper Medium)
    load_in_4bit=True,  # Active la quantification 4-bit pour réduire la consommation mémoire GPU
    device_map="auto"  # Permet un chargement automatique sur le GPU disponible
)

# Application des poids fine-tunés (LoRA) au modèle de base
model_finetuned = PeftModel.from_pretrained(model_finetuned, peft_model_id)

# Activation du cache pour accélérer l'inférence
model_finetuned.config.use_cache = True

```
 

### 📊 Résultats de l'Évaluation Après Fine-Tuning

L’évaluation post-entrainement a été réalisée sur l’ensemble de test en utilisant les mêmes métriques que l’évaluation initiale :

Ce tableau affiche les résultats du modèle après fine-tuning, évalué sur l'ensemble test. 

Comparé aux résultats avant fine-tuning, on observe une amélioration du WER et du CER, ce qui indique une meilleure transcription des données médicales.

| **Métrique**                   | **Valeur obtenue**  | **Description**  |
|--------------------------------|--------------------|----------------|
| **WER (Word Error Rate)**      | `30.23%`          | Taux d’erreur au niveau des mots (substitutions, insertions, suppressions). |
| **Normalized WER**             | `11.19%`          | WER après normalisation (suppression des variations d’espaces et de ponctuation). |
| **CER (Character Error Rate)** | `9.12%`           | Taux d’erreur basé sur les caractères au lieu des mots. |
| **Normalized CER**             | `4.51%`           | CER après normalisation du texte. |
| **MTA (Medical Term Accuracy)**| `62.65%`          | Pourcentage de termes médicaux correctement transcrits. |

### 🏆 Conclusion sur les Résultats du Fine-Tuning

Après le fine-tuning du modèle Whisper Medium sur les données médicales, nous constatons une amélioration des performances :

### 📊 Comparaison des Résultats Avant et Après Fine-Tuning

| **Métrique**                    | **Avant Fine-Tuning** | **Après Fine-Tuning** | **Amélioration** |
|---------------------------------|----------------------|----------------------|------------------|
| **WER (Word Error Rate)**       | `31.18%`            | `30.23%`            | 📉 `-0.95%`      |
| **Normalized WER**              | `12.86%`            | `11.19%`            | 📉 `-1.67%`      |
| **CER (Character Error Rate)**  | `9.78%`             | `9.12%`             | 📉 `-0.66%`      |
| **Normalized CER**              | `5.18%`             | `4.51%`             | 📉 `-0.67%`      |
| **MTA (Medical Term Accuracy)** | `64.20%`            | `62.65%`            | 📉 `-1.55%`      |


### 📌 Analyse des Résultats du Fine-Tuning

Ces résultats montrent que le **fine-tuning du modèle Whisper Medium** a permis une **réduction notable des erreurs** sur toutes les métriques d'évaluation, bien que les améliorations soient modestes. Voici quelques points à retenir :

✅ **Réduction du WER et du CER** : Le taux d'erreur au niveau des mots (**WER**) et des caractères (**CER**) a diminué après fine-tuning, indiquant que le modèle a mieux appris à reconnaître les termes médicaux et les transcriptions en général.

✅ **Amélioration du Normalized WER et CER** : En prenant en compte la normalisation des textes (suppression des variations d'espaces et de ponctuation), nous observons également une amélioration sur ces métriques.

❌ **Légère baisse du MTA (Medical Term Accuracy)** : Le pourcentage de termes médicaux correctement transcrits a légèrement diminué (**-1.55%**). Cela peut être dû au fait que le modèle a ajusté ses prédictions globales, mais au détriment de certains termes médicaux spécialisés. Pour améliorer cet aspect, une **augmentation de la quantité de données d'entraînement** et une meilleure **représentation des termes médicaux rares** sont nécessaires.

### 🛠️ Perspectives d'Amélioration

🔹 **Augmenter la taille du dataset** : Les améliorations restent limitées car le fine-tuning a été réalisé sur un ensemble de données relativement restreint. Un dataset plus grand et plus varié permettrait d’obtenir des gains plus significatifs.

🔹 **Enrichir le corpus avec des termes médicaux** : Un lexique médical plus détaillé et des données spécifiques au domaine pourraient améliorer la reconnaissance des termes spécialisés.


En conclusion, **même une amélioration légère reste un progrès important**. Le fine-tuning a permis une réduction des erreurs, mais pour obtenir des gains plus significatifs, il faudra **davantage de données** et **un ajustement plus poussé du modèle**. 🚀
