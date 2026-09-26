# Emotion Classification Using RNN, LSTM, GRU, and BiGRU

## 1. Introduction

This project is a natural language processing (NLP) system that reads an English sentence and predicts the emotion expressed in that sentence. The project compares four recurrent neural network approaches:

- Simple RNN
- LSTM
- GRU
- Bidirectional GRU (BiGRU)

The final selected model is a BiGRU network. It achieved **92.00% test accuracy** on the recorded notebook run.

The main implementation is in [train.model.ipynb](train.model.ipynb). The trained model is stored in [Artifacts/BiGRU_Model.keras](Artifacts/BiGRU_Model.keras). The workspace also contains [Artifacts/BiGRU_Model.h5](Artifacts/BiGRU_Model.h5).

## 2. Problem Being Solved

People express emotions through words, but the same emotion can be written in many different ways. A rule-based system would need a large list of words and would struggle with sentence context. This project solves the problem by learning patterns from labelled text.

Given a sentence such as:

> I feel so sad and lonely after hearing the bad news.

the system predicts one of six emotion classes: `sadness`, `joy`, `love`, `anger`, `fear`, or `surprise`.

Potential uses include sentiment and emotion dashboards, customer feedback analysis, social media monitoring, support-ticket prioritisation, and educational NLP demonstrations.

## 3. Dataset

The project downloads the `dair-ai/emotion` dataset from Hugging Face using `load_dataset`.

| Split | Number of records |
|---|---:|
| Train | 16,000 |
| Validation | 2,000 |
| Test | 2,000 |

The notebook uses the train and test data directly in the model training calls. The dataset defines six labels:

| Label | Training records |
|---|---:|
| joy | 5,362 |
| sadness | 4,666 |
| anger | 2,159 |
| fear | 1,937 |
| love | 1,304 |
| surprise | 572 |

The classes are imbalanced: `joy` has the most examples and `surprise` has the fewest. To reduce the effect of this imbalance, the notebook calculates balanced class weights with `sklearn.utils.class_weight.compute_class_weight` and supplies them during training.

## 4. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Interactive development, training, and output presentation |
| NumPy | Numerical arrays and label conversion |
| pandas | DataFrame creation and result tables |
| Hugging Face Datasets | Downloading and loading the emotion dataset |
| TensorFlow / Keras | Building, training, and saving neural networks |
| scikit-learn | Balanced class weights and confusion matrix |
| Matplotlib and Seaborn | Data visualisation and confusion-matrix heatmap |
| pickle | Saving the fitted tokenizer |

The recorded notebook environment used TensorFlow `2.20.0`, Keras `3.13.1`, pandas `2.2.3`, NumPy `2.4.1`, scikit-learn `1.7.2`, seaborn `0.13.2`, and datasets `5.0.1`.

## 5. Processing Pipeline

1. Download the dataset from Hugging Face.
2. Separate text and numeric labels for the train and test splits.
3. Map numeric labels to readable names for analysis.
4. Create pandas DataFrames containing `text` and `label`.
5. Check the class distribution and missing values.
6. Fit a Keras `Tokenizer` on the training text.
7. Limit the vocabulary to `10,000` words and represent unknown words with `<unk>`.
8. Convert sentences into integer token sequences.
9. Pad or truncate every sequence to `50` tokens using post-padding.
10. Calculate balanced class weights.
11. Train and evaluate RNN, LSTM, GRU, and BiGRU models.
12. Use the best model to classify new sample sentences.
13. Save the trained BiGRU model and tokenizer artifacts.

## 6. Model Architectures

All models use the same general structure so that their results can be compared fairly:

`Embedding -> recurrent layers -> Dropout -> Dense softmax output`

### Simple RNN

- Embedding size: 128
- SimpleRNN layer: 128 units, returns sequences
- SimpleRNN layer: 64 units
- Dropout: 0.5 after each recurrent stage

### LSTM

- Embedding size: 128
- LSTM layer: 128 units, returns sequences
- LSTM layer: 64 units
- Dropout: 0.5 after each recurrent stage

### GRU

- Embedding size: 128
- GRU layer: 128 units, returns sequences
- GRU layer: 64 units
- Dropout: 0.5 after each recurrent stage

### BiGRU

- Embedding size: 128
- Bidirectional GRU layer: 128 units, returns sequences
- Bidirectional GRU layer: 64 units
- Dropout: 0.5 after each recurrent stage
- Six-unit Dense output layer with softmax activation

The BiGRU reads each sequence in both directions. This allows it to use information from words before and after the current position. For example, the meaning of a word can depend on the context that follows it as well as the context that precedes it.

## 7. Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Metric | Accuracy |
| Maximum epochs | 20 |
| Batch size | 32 |
| Validation data | `padded_test_seqence`, `test_labels` |
| Class balancing | Balanced class weights |
| Early stopping monitor | `val_loss` |
| Early stopping patience | 3 epochs |
| Best weights | Restored after early stopping |

The notebook uses early stopping to stop training when validation loss no longer improves and restores the best validation weights.

## 8. Model Evaluation Results

The following values are from the executed notebook cells.

| Model | Test loss | Test accuracy |
|---|---:|---:|
| GRU | 1.774120 | 34.45% |
| LSTM | 1.796845 | 11.15% |
| Simple RNN | 1.784438 | 10.30% |
| **BiGRU** | **0.209887** | **92.00%** |

### Difference Between the Models

- **Simple RNN:** The simplest recurrent model. It is lightweight but can lose useful information over longer sequences because of the vanishing-gradient problem.
- **LSTM:** Uses four gates to control information flow and can preserve long-term information. It is more complex and computationally heavier than a GRU.
- **GRU:** Uses fewer gates than an LSTM, so it is generally faster and has fewer parameters. In this run it performed better than the standalone RNN and LSTM.
- **BiGRU:** Uses GRU layers in both forward and backward directions. It receives context from the complete sentence and produced the strongest result in this experiment.

The BiGRU improved test accuracy by **57.55 percentage points** over the standalone GRU and by **81.70 percentage points** over the Simple RNN. Its test loss was also much lower, which indicates better probabilistic predictions on the test set.

Accuracy should be interpreted together with the confusion matrix because the dataset is imbalanced. A high overall accuracy does not guarantee equal performance for every emotion class.

## 9. Screenshots and Visual Outputs

The notebook contains the following visual outputs. They are embedded in the executed notebook rather than exported as separate image files in the current workspace.

1. **Class-distribution chart:** The Seaborn count plot shows the number of training examples for each emotion. It makes the class imbalance visible.
2. **Model comparison table:** The notebook displays the test loss and test accuracy for RNN, LSTM, and GRU before the BiGRU is trained.
3. **BiGRU confusion matrix:** The Seaborn heatmap compares the true emotion labels with the BiGRU predictions for all six classes.
4. **Prediction output:** The notebook prints ten sample sentences and their predicted emotions.
5. **Model summary:** Keras prints the layers and parameter information for the BiGRU architecture.

To capture these screenshots, open [train.model.ipynb](train.model.ipynb), run the notebook, and capture the output below the EDA count plot, the model comparison table, the confusion-matrix heatmap, and the sample prediction cell. Keeping the screenshots beside this document makes the report suitable for a presentation or project submission.

## 10. Sample Predictions

The executed notebook produced these example results:

| Input summary | Predicted emotion |
|---|---|
| Feeling happy because everything went perfectly | joy |
| Feeling sad and lonely after bad news | sadness |
| Angry because a request was ignored | anger |
| Terrified by a strange noise | fear |
| Surprised by a wonderful gift | surprise |
| Joyful after achieving a dream | joy |
| Worried and scared about tomorrow | fear |
| Excited about going on a trip | anger |
| Disappointed by an unexpected event result | sadness |
| Shocked by unexpected news | surprise |

Most examples match their expected emotion. The sentence about being excited for a trip was predicted as `anger`, showing that the model can still confuse emotions with similar or insufficiently represented language.

## 11. Saved Artifacts

The notebook creates an `Artifacts` directory and saves:

- `BiGRU_Model.keras`: Keras model file containing the final BiGRU architecture and learned weights.
- `tokenizer.pkl`: Fitted tokenizer required to convert new text into the same integer representation used during training.

The workspace also contains `BiGRU_Model.h5`, an HDF5-format model file. For new Keras projects, the `.keras` format is the primary saved model produced by the notebook.

The tokenizer is essential for inference. A new sentence must use the original tokenizer, the same post-padding strategy, and `maxlen=50`; otherwise the model receives data in a different representation from the one used during training.

## 12. Limitations and Improvements

- The dataset contains short English text, so performance may decrease on longer, sarcastic, multilingual, or domain-specific text.
- The classes are imbalanced even though class weights are used.
- The report records accuracy and loss, but per-class precision, recall, and F1-score should also be calculated for a fuller evaluation.
- The sample prediction set is small and is not a substitute for a separate real-world test set.
- The notebook uses the test split as `validation_data` during training, so a stricter experiment should keep the official test set untouched and use the dataset validation split for early stopping.
- Reproducibility would improve by setting random seeds and recording the exact Python environment in a requirements file.

## 13. Conclusion

This project demonstrates an end-to-end emotion classification workflow, from downloading and exploring labelled text to preprocessing, comparing recurrent architectures, evaluating predictions, visualising errors, and exporting a trained model. The BiGRU was selected because its bidirectional context produced the best recorded result: **0.209887 test loss and 92.00% test accuracy**.
