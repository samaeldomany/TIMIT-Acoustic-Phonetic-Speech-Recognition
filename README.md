# TIMIT-Acoustic-Phonetic-Speech-Recognition

Overview
--------

This repository contains a PyTorch implementation of an Automatic Speech Recognition (ASR) system focused on phoneme recognition. It trains a Recurrent Neural Network (RNN) on the DARPA TIMIT Acoustic-Phonetic Continuous Speech dataset. The model leverages Bidirectional LSTMs and is trained using Connectionist Temporal Classification (CTC) loss to align variable-length audio features with phoneme sequences.

Key Features
------------

*   **Automated Data Handling**: Automatically fetches the TIMIT dataset from Kaggle using kagglehub, with a built-in fallback to generate synthetic data if the download fails.
    
*   **Audio Preprocessing**: Converts raw audio waveforms into 13-dimensional Mel-Frequency Cepstral Coefficients (MFCCs) at a 16kHz sample rate using torchaudio.
    
*   **Robust Architecture**: Implements a PyTorch-based RecurrentPhonemeModel utilizing Layer Normalization, Bidirectional LSTMs, and Dropout for regularization.
    
*   **Custom Decoders**: Includes both **Greedy Decoding** and **Beam Search Decoding** algorithms to transcribe the CTC network's output probabilities.
    
*   **Performance Metrics**: Evaluates model accuracy using Levenshtein distance to compute the Phoneme Error Rate (PER).
    

Requirements
------------

Ensure you have Python installed, along with the following dependencies:

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pip install torch torchaudio numpy matplotlib soundfile kagglehub   `

Model Architecture
------------------

The core model (RecurrentPhonemeModel) is designed to process sequential MFCC features and output phoneme probabilities:

*   **Input**: 13-dim MFCC features.
    
*   **Normalization**: nn.LayerNorm applied to the input features.
    
*   **RNN**: 2-layer Bidirectional LSTM with a hidden size of 256 and a dropout rate of 0.2.
    
*   **Classifier**: A linear layer mapping the LSTM outputs to 48 classes (47 TIMIT phonemes + 1 token for CTC).
    

Training Details
----------------

The model is configured with the following hyperparameters:

*   **Batch Size**: 8
    
*   **Optimizer**: AdamW
    
*   **Learning Rate**: 5e-4
    
*   **Epochs**: 10
    
*   **Loss Function**: nn.CTCLoss (with blank token at index 0)
    
*   **Hardware**: Automatically utilizes CUDA (GPU) if available, otherwise falls back to CPU.
    

Results
-------

Over the course of 10 epochs, the training loss steadily decreases from approximately **2.47** to **0.89**.

During evaluation on the test set, the model successfully aligns audio frames to phonemes and decodes the sequences. The current configuration achieves a **Mean Phoneme Error Rate (PER) of ~0.34**. The notebook also includes a visualization tool that plots the CTC posteriors (alignment heatmaps) to help analyze the model's confidence across time steps.
