# LIAR2 Fake News Detection

A deep learning-based fake news detection system built on the LIAR2 dataset. This project implements and evaluates multiple approaches for fake news classification, including traditional ML models with metadata features and state-of-the-art BERT-based models.

## 📊 Project Overview

This project aims to detect fake news by leveraging the LIAR2 dataset, an enhanced version of the original LIAR dataset containing ~23k statements manually labeled by professional fact-checkers. The system classifies statements into six veracity categories:

- 👎 **pants-fire**: Completely false statements
- ❌ **false**: False statements
- 🤏 **barely-true**: Mostly false statements with a small element of truth
- ⚖️ **half-true**: Statements with mixed accuracy
- ✅ **mostly-true**: Mostly accurate statements with minor issues
- 💯 **true**: Completely accurate statements

## 🚀 Features

- **Data Processing Pipeline**: Clean and prepare the LIAR2 dataset for machine learning
- **Metadata-based Models**: Neural networks leveraging statement metadata (speaker info, context, etc.)
- **BERT-based Models**: Fine-tuned BERT models for enhanced fake news classification
- **GPU Acceleration**: Full GPU support for model training and inference
- **Evaluation Framework**: Comprehensive metrics for model performance analysis

## 🗂️ Project Structure

```
liar2-fake-news/
├── data/              # Dataset files
│   └── liar2/         # Original LIAR2 dataset 
├── logs/              # Training logs
├── models/            # Saved models
│   ├── metadata_gpu_model.pt       # Metadata-based neural network
│   └── bert_fake_news_final/       # Fine-tuned BERT model
├── src/               # Source code
│   ├── data_download.py            # Dataset download script
│   ├── data_prep.py                # Data preprocessing
│   ├── train_meta_baseline.py      # Metadata-based model training
│   ├── train_bert.py               # BERT model training
│   └── test_bert.py                # BERT model testing script
└── requirements.txt    # Python dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/madboy482/FakeNewsDetection.git
   cd FakeNewsDetection
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download and prepare the dataset:
   ```
   python src/data_download.py
   python src/data_prep.py
   ```

## 📝 Usage

### Training Models

Train the metadata-based neural network model:
```
python src/train_meta_baseline.py
```

Train the BERT-based model (requires GPU):
```
python src/train_bert.py
```

### Testing the BERT Model

Test the BERT model with sample statements:
```
python src/test_bert.py
```

## 📈 Performance

Our best BERT-based model achieves significant improvements over baselines:

| Model | Accuracy | Macro F1 | 
|-------|----------|----------|
| Metadata NN | 51.19% | 0.48 |
| BERT | ~65% | ~0.63 |

## 🔍 Future Work

- Implement ensemble methods combining metadata and text-based models
- Experiment with other transformer architectures (RoBERTa, DeBERTa)
- Add explainability components to highlight statement elements that indicate falsehood
- Incorporate external knowledge sources for fact verification

## 📚 Citation

If you use the LIAR2 dataset, please cite the original paper:

```
@article{cheng2024enhanced,
  title={An Enhanced Fake News Detection System With Fuzzy Deep Learning},
  author={Cheng, Xu and Liu, Weiwei and Wang, Yue and Tang, Bo and He, Yingchun},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
