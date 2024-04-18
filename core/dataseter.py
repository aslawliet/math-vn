from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from core.multipack_sampler import MultipackDistributedBatchSampler
from dataclasses import dataclass
import os, datasets, logging, torch, transformers, math
from typing import Counter, Dict, Sequence
from torch.utils.data import Dataset
import torch.distributed as dist
import numpy as np

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
DEFAULT_UNK_TOKEN = "<|unk|>"

def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    train_on_inputs: bool,
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    sources = [f"{question}" for question in samples["input"]]
    examples = [s for s in sources]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = [float(torch.tensor(label)) for label in samples["label"]]

    return dict(input_ids=input_ids, labels=labels)

def _filter_tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    samples = []
    for text in strings:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        if tokens.input_ids.squeeze().numel() < tokenizer.model_max_length:
            samples.append(True)
        else:
            samples.append(False)

    return samples

def filter_long_samples(
    samples: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    sources = [f"{question}" for question in samples["input"]]
    examples = [s for s in sources]

    return _filter_tokenize_fn(examples, tokenizer)

class OVMDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        train_on_inputs: bool,
        tokenizer: transformers.PreTrainedTokenizer,
        data_paths,
        limit=None,
    ):
            super(OVMDataset, self).__init__()
        # try:
            workers = math.ceil(os.cpu_count() / dist.get_world_size())
            logging.warning(f"TOKENIZING WITH NUM_WORKERS: {workers}")
            dataset = (
                datasets.load_dataset(
                    "json",
                    data_files=data_paths,
                    split=f"train[0:{limit}]" if limit else "train",
                )
                .filter(
                    lambda samples: filter_long_samples(samples, tokenizer),
                    batched=True,
                    batch_size=3000,
                    num_proc=workers,
                )
                .map(
                    lambda samples: preprocess(train_on_inputs, samples, tokenizer),
                    batched=True,
                    batch_size=3000,
                    num_proc=workers,
                )
            )

            self.input_ids = dataset["input_ids"]
            self.labels = dataset["labels"]
            # Calculate pos_weights and neg_weights
            label_counts = Counter(self.labels)

            total_samples = len(self.labels)
            self.pos_weights =   label_counts[1]/total_samples if label_counts[1] > 0 else 0.0
            self.neg_weights =   label_counts[0]/total_samples if label_counts[0] > 0 else 0.0


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=torch.tensor(self.input_ids[i]),
            labels=torch.tensor(self.labels[i]),
        )  
      
@dataclass
class OVMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        
        labels = torch.tensor(labels)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        
def get_dataloader(
    use_multipack_sampler, max_length, dataset, world_size, local_rank, shuffle, seed, collator, batch_size):
    if use_multipack_sampler:
        lengths = np.array([len(tokens["input_ids"]) for tokens in dataset])
        sampler = MultipackDistributedBatchSampler(
            batch_max_length=batch_size * max_length, lengths=lengths, num_replicas=world_size,
            rank=local_rank, seed=seed
        )

        loader = DataLoader(dataset, pin_memory=True, collate_fn=collator, batch_sampler=sampler)
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=shuffle, seed=seed
        )

        loader = DataLoader(
            dataset, shuffle=False, pin_memory=True, drop_last=True, batch_size=batch_size,
            collate_fn=collator, sampler=sampler
        )

    return sampler, loader
