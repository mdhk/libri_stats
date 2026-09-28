from argparse import ArgumentParser
from datasets import load_dataset
from collections import Counter
import pandas as pd


def get_word_counts(libri_lengths_split):
    all_text = ' '.join(libri_lengths_split['text'])
    word_counts = pd.DataFrame(
        Counter(all_text.split(' ')).items(),
        columns=['word', 'count']
    ).sort_values(by='count', ascending=False)
    return word_counts

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--HF_TOKEN",
        required=True,
        type=str,
        help="HuggingFace token"
    )
    parser.add_argument(
        "--HF_REPO",
        type=str,
        default="mariannedhk/librispeech_lengths",
        help="HuggingFace hub repository"
    )
    args = parser.parse_args()

    libri_lengths = load_dataset(args.HF_REPO, token=args.HF_TOKEN)

    for split in libri_lengths.keys():
        split_word_counts = get_word_counts(libri_lengths[split])
        split_word_counts.to_csv(f'librispeech_{split}_wordcounts.csv', index=False)