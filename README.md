# Traduction Multilingue de Vidéos — Pipeline Deep Learning

> Projet académique de Deep Learning — Comparaison d'architectures NMT pour le sous-titrage et le doublage automatique de vidéos françaises.

---

## Aperçu

Ce projet construit un **pipeline complet de traduction vidéo** :  
`Vidéo source (FR)` → **ASR** → **Traduction NMT** → **TTS** → `Vidéo doublée + sous-titrée`

Il compare quatre approches de traduction automatique (NMT) sur le corpus **mTEDx français**, avec évaluation quantitative (BLEU, chrF) et analyse qualitative.

---

## Pipeline

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  Vidéo .mp4 │───▶│  ASR — Whisper   │───▶│  Segments horodatés  │
└─────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                        │
                          ┌─────────────────────────────▼──────────────────────────────┐
                          │              Traduction NMT (FR → EN / ES / IT / PT)        │
                          │  ┌─────────────┐  ┌──────────┐  ┌────────────┐  ┌──────┐  │
                          │  │  MarianMT   │  │ NLLB-200 │  │ Gemma3:27b │  │ LSTM │  │
                          │  └─────────────┘  └──────────┘  └────────────┘  └──────┘  │
                          └─────────────────────────────┬──────────────────────────────┘
                                                        │
                     ┌──────────────────────────────────▼───────────────────────────────┐
                     │                           Sorties                                 │
                     │   ├── Fichiers .SRT / .VTT (sous-titres)                         │
                     │   ├── Audio TTS (edge-tts)                                        │
                     │   └── Vidéo finale (.mp4) — piste audio doublée + sous-titres    │
                     └──────────────────────────────────────────────────────────────────┘
```

---

## Approches comparées

| # | Modèle | Type | Points forts |
|---|--------|------|-------------|
| A | [MarianMT](https://huggingface.co/Helsinki-NLP) (Helsinki-NLP) | Transformer spécialisé par paire | Rapide, précis sur paires couvertes |
| B | [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) (Meta) | Transformer multilingue massif | Un seul modèle pour toutes les langues |
| C | Gemma3:27b via Ollama | LLM généraliste | Flexible par prompting, sans API cloud |
| D | Seq2Seq LSTM (custom) | RNN entraîné from scratch | Baseline pédagogique |

**ASR :** [Whisper](https://huggingface.co/openai/whisper-medium) (OpenAI) — medium/small selon GPU disponible  
**TTS :** [edge-tts](https://github.com/rany2/edge-tts) (Microsoft Neural TTS)

---

## Évaluation

Les modèles sont évalués sur des segments du corpus mTEDx avec :

- **BLEU** — recouvrement en n-grammes
- **chrF** — comparaison au niveau caractères (robuste sur phrases courtes)
- **WER / CER** — qualité de la transcription ASR
- Analyse qualitative : contresens, omissions, hallucinations, noms propres, effet cascade ASR

---

## Structure du projet

```
.
├── src/
│   └── notebook_benchmark_traduction.ipynb   # Notebook principal
├── data/
│   └── inputs/test.mp4                        # Vidéo de démonstration
├── outputs/                                   # SRT, TTS, vidéos générées
├── doc/
│   └── Projet_Traduction_Video_Multilingue_DL.pdf
├── requirements.txt
└── environment.yml
```

---

## Installation

**Avec Conda (recommandé) :**
```bash
conda env create -f environment.yml
conda activate tradl
```

**Avec pip :**
```bash
pip install -r requirements.txt
```

> Nécessite Python 3.10 et PyTorch ≥ 2.2. GPU CUDA recommandé pour Whisper et NLLB.

---

## Utilisation

Ouvrir le notebook principal :
```bash
jupyter notebook src/notebook_benchmark_traduction.ipynb
```

Le notebook est structuré en sections indépendantes :
1. **Config** — chemins, langues cibles, seed de reproductibilité
2. **ASR** — transcription Whisper + évaluation WER/CER
3. **Traduction** — 4 approches comparées (A, B, C, D)
4. **Évaluation** — BLEU/chrF, benchmarks latence/RAM, graphiques
5. **Pipeline vidéo** — traitement complet de `test.mp4` → vidéo doublée
6. **Conclusion**

---

## Corpus

**[mTEDx](http://www.openslr.org/100/)** — corpus multilingue de conférences TED, fournissant des paires parallèles FR/EN, FR/ES, FR/IT, FR/PT avec transcriptions et alignements temporels.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-orange?logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)

`transformers` · `whisper` · `MarianMT` · `NLLB-200` · `edge-tts` · `sacrebleu` · `jiwer` · `ffmpeg`

---

## Auteur

**Mathias Barrois** — [barrois.mathias@gmail.com](mailto:barrois.mathias@gmail.com)
