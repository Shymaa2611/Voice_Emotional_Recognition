# Voice Emotional Recognition Using OpenAI Whisper Model

## Model Details
utilize the Whisper model from OpenAI for feature extraction, which is known for its robust speech recognition capabilities.On top of Whisper, we add a classifier to distinguish between different emotional tones.The model is fine-tuned on the labeled dataset of emotional tones.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Usage](#usage)
- [Checkpoint](#checkpoint)

## Overview

This project aims to build an emotional recognition system from voice data by fine-tuning OpenAI's Whisper model. The goal is to classify various emotions like happiness, sadness, anger, and neutrality based on speech recordings. The project leverages deep learning techniques and the power of Whisper, a state-of-the-art speech recognition model, to extract features from voice and map them to specific emotional labels.

## Requirements

    Python 3.7 or higher
    PyTorch
    Transformers library
You can install the required packages using pip:
``` bash
!pip install -r requirements.txt
```

## Dataset

#### Toronto emotional speech set (TESS)
There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.
The dataset is organised such that each of the two female actor and their emotions are contain within its own folder. And within that, all 200 target words audio file can be found. The format of the audio file is a WAV format 
can download dataset from : https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

## Installation

Clone the repository and navigate into the project directory:
```bash
git clone https://github.com/Shymaa2611/Voice_Emotional_Recognition.git
cd Voice_Emotional_Recognition
```

## Training

To fine-tune the Whisper model, run the following command:
``` bash 
!python main.py

```


## Usage

``` python



```


## Checkpoint
can download from :