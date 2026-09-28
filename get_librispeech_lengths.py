from argparse import ArgumentParser
from datasets import load_dataset

def add_summary_columns(example):
    example['audio_duration'] = example['audio'].metadata.duration_seconds
    example['word_count'] = len(example['text'].split(' '))
    return example

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--HF_TOKEN",
        required=True,
        type=str,
        help="HuggingFace token"
    )
    parser.add_argument(
        "--LIBRISPEECH_PATH",
        type=str,
        default="/Volumes/Datasets/hf_librispeech",
        help="path to LibriSpeech dataset (hf_librispeech)"
    )
    parser.add_argument(
        "--HF_REPO",
        type=str,
        default="mariannedhk/librispeech_lengths",
        help="HuggingFace hub repository"
    )
    args = parser.parse_args()

    print('Loading LibriSpeech dataset...')
    librispeech = load_dataset(args.LIBRISPEECH_PATH)

    print('Adding summary columns...')
    librispeech = librispeech.map(add_summary_columns)
    libri_lengths = librispeech.remove_columns(['file', 'audio'])
    
    print('Uploading to HF hub...')
    libri_lengths.push_to_hub("mariannedhk/librispeech_lengths", token=args.HF_TOKEN)

    print('Done!')