# src/data_prep.py

import pandas as pd
import re, string, nltk
from nltk.corpus import stopwords

def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z ]', '', str(text).lower())
    tokens = [w for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(tokens)

def main():
    nltk.download('stopwords')
    df = pd.read_csv("data/liar2_full.csv")

    # Clean key text fields
    df['clean_statement'] = df['statement'].map(clean_text)
    df['clean_justification'] = df['justification'].map(clean_text)

    # Prepare metadata columns
    meta_cols = [
        'label',
        'clean_statement',
        'clean_justification',
        'subject',
        'speaker',
        'speaker_description',
        'state_info',
        'true_counts',
        'mostly_true_counts',
        'half_true_counts',
        'mostly_false_counts',
        'false_counts',
        'pants_on_fire_counts',
        'context'
    ]
    df_meta = df[meta_cols].copy()

    df_meta.to_csv("data/liar2_meta.csv", index=False)
    print("✅ Saved ’liar2_meta.csv’ with shape", df_meta.shape)
    print("Columns included:", df_meta.columns.tolist())

if __name__ == "__main__":
    main()
