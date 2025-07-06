"""
Using an NLP Data Loader to translate Spanisch <-> English (Multi30k).

# Creating an NLP Data Loader
As an AI engineer working on a cutting-edge language translation project, you are tasked with bridging the communication gap between speakers of different languages. Translating languages is no small feat, especially given the intricacies, nuances, and cultural contexts embedded within them. Central to the success of this endeavor is the data - large corpora of bilingual sentences that serve as the bedrock of your models.
In PyTorch, the data loader plays an indispensable role in managing this vast amount of data. For natural language processing (NLP) tasks like yours, data often comes in variable lengths due to differing sentence structures and lengths across languages. The data loader efficiently batches these variable-length sequences, ensuring that your models are trained on diverse examples in every iteration. This batching is crucial for harnessing the power of parallel computation on GPUs, thus expediting the training process.
Furthermore, the data loader aids in shuffling the data set, which is vital for preventing models from memorizing the sequence of training data and promoting better generalization. Especially for NLP tasks, where data might be ordered or clustered by topics, shuffling ensures that the model remains robust and doesn't develop biases based on the order of input.
Lastly, in the world of NLP, preprocessing steps such as tokenization, padding, and numericalization are paramount. The data loader in PyTorch provides hooks that allow us to seamlessly integrate these preprocessing steps, ensuring that the raw textual data is transformed into a format that's amenable for deep learning models.
In PyTorch, the data loader plays an indispensable role in managing this vast amount of data.
In this lab, you will cover the whole process of loading and collating text data using PyTorch.
## Installing required libraries
"""
!pip install nltk
!pip install transformers==4.42.1
!pip install sentencepiece
!pip install spacy
!pip install numpy==1.26.0
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
!pip install torch==2.2.2 torchtext==0.17.2
!pip install torchdata==0.7.1
!pip install portalocker
!pip install numpy pandas
!pip install numpy scikit-learn

"""
You can check the installed version of each package to make sure you have set the right environment.
## Importing required libraries
"""
import torchtext
print(torchtext.__version__)
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Mapper
import torchtext

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random

sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]
# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# Create an instance of your custom dataset
custom_dataset = CustomDataset(sentences)
# Define batch size
batch_size = 2
# Create a DataLoader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
# Iterate through the DataLoader
for batch in dataloader:
    print(batch)

"""As shown above, the data is organized into batches of 2 sentences each. It's important to note that deep learning models can only comprehend numerical data, and words are meaningless to them. Therefore, your next step is to convert these sentences into tensors. Let's see how to do this.
"""
sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]
# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)
# Tokenizer
tokenizer = get_tokenizer("basic_english")
# Build vocabulary
vocab = build_vocab_from_iterator(map(tokenizer, sentences))
# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)
print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")

"""
Please go ahead and uncomment the following code to apply the data loader and observe the results:
"""
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.sentences[idx])
        # Convert tokens to tensor indices using vocab
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)
# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)
# Define batch size
batch_size = 2
# Create a data loader
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
# Iterate through the data loader
for batch in dataloader:
    print(batch)

"""
### Custom collate function
"""
# Create a custom collate function
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

"""
Let's see how it actually affects the shape of curated batches. First, you create a data loader from a collate function with `batch_first=True`:
"""
# Create a data loader with the custom collate function with batch_first=True,
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)
# Iterate through the data loader
for batch in dataloader:
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)

"""
Now, you can try `batch_first=False` which is the DEFAULT value:
"""
# Create a custom collate function
def collate_fn_bfFALSE(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, padding_value=0)
    return padded_batch

"""
Now, you look into the curated data:
"""
# Create a data loader with the custom collate function with batch_first=True,
dataloader_bfFALSE = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn_bfFALSE)
# Iterate through the data loader
for seq in dataloader_bfFALSE:
    for row in seq:
        #print(row)
        words = [vocab.get_itos()[idx] for idx in row]
        print(words)

"""
It can be seen that the first dimension is now the sequence instead of batch, which means sentences will break so that each row includes a token from each sequence. For example the first row, (['if', 'fame']), includes the first tokens of all the sequences in that batch. You need to be aware of this standard to avoid any confusion when working with recurrent neural networks (RNNs) and transformers.
"""
# Iterate through the data loader with batch_first = TRUE
for batch in dataloader:
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])

"""
You will see that each batch has a fixed size for all the sequences within the batch.
"""
# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

custom_dataset=CustomDataset(sentences)

"""
You have the raw text:
"""
custom_dataset[0]

"""
You create the new ```collate_fn```
"""
def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)

    # Return the padded batch
    return padded_batch

"""
Create a data loader with the custom collate function.
"""
# Create a data loader for the custom dataset
dataloader = DataLoader(
    dataset=custom_dataset,   # Custom PyTorch Dataset containing your data
    batch_size=batch_size,     # Number of samples in each mini-batch
    shuffle=True,              # Shuffle the data at the beginning of each epoch
    collate_fn=collate_fn      # Custom collate function for processing batches
)

"""
You will see that the result is a tensor of the same shape for each sample in the batch.
"""
for batch in dataloader:
    print(batch)
    print("shape of sample",len(batch))

"""
As a result, batches of tensors with equal lengths have been successfully created.
## Exercise
Create a data loader with a collate function that processes batches of French text (provided below). Sort the data set on sequences length. Then tokenize, numericalize and pad the sequences. Sorting the sequences will minimize the number of `<PAD>`tokens added to the sequences, which enhances the model's performance. Prepare the data in batches of size 4 and print them.
"""
!python -m spacy download fr_core_news_sm
corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !"
]
# This is the batch size.
batch_size = 4
# Choose the tokenizer and build the vocab for the current words.
tokenizer = get_tokenizer('spacy', language='fr_core_news_sm')
vocab = build_vocab_from_iterator(map(tokenizer, corpus))
# Create an instance of your custom data set.
sorted_corpus = sorted(corpus, key=lambda s: len(tokenizer(s)))
custom_dataset = CustomDataset(sorted_corpus)
# The collate function
def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)

    # Return the padded batch
    return padded_batch
# Creates the Data Loader
dataloader = DataLoader(
    dataset=sorted_corpus,   # Custom PyTorch Dataset containing your data
    batch_size=batch_size,     # Number of samples in each mini-batch
    shuffle=False,              # Shuffle the data at the beginning of each epoch definitely not!
    collate_fn=collate_fn      # Custom collate function for processing batches
)

for batch in dataloader:
    print(batch)

"""
##  Data loader for German-English translation task
This section sets the stage for German-English machine translation using the torchtext and spaCy libraries. It adjusts data set URLs for the Multi30k data set, configures tokenizers for both languages, and establishes vocabularies with special tokens. This foundation is crucial for building and training effective translation models.
### Translation data set
In this section, you fetch a language translation data set called Multi30k. You will modify its default training and validation URLs, and then retrieve and print the first pair of German-English sentences from the training set. First, you will override the default URLs:
"""
# You would modify the URLs for the data set since the links to the original data set are broken
multi30k.URL["train"] = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/training.tar.gz"
multi30k.URL["valid"] = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/validation.tar.gz"

"""
Define the source language as German ('de') and target language as English ('en'). In Python, global variables are variables defined outside of a function, accessible both inside and outside of functions. They are often written in all caps as a convention to indicate they are constant, global nature and to differentiate them from regular variables.
"""
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

"""
Initialize the training data iterator for the Multi30k dataset with the specified source and target languages:
"""
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))

"""
Create an iterator for the training data set:
"""
data_set = iter(train_iter)

"""
You can print out the first five pairs of source and target sentences from the training data set:
"""
for n in range(5):
    # Getting the next pair of source and target sentences from the training data set
    src, tgt = next(data_set)

    # Printing the source (German) and target (English) sentences
    print(f"sample {str(n+1)}")
    print(f"Source ({SRC_LANGUAGE}): {src}\nTarget ({TGT_LANGUAGE}): {tgt}")

"""
### Tokenizer setup
"""
german, english = next(data_set)
print(f"Source German ({SRC_LANGUAGE}): {german}\nTarget English  ({TGT_LANGUAGE}): { english }")

"""
Import the```get_tokenizer``` utility function from ```torchtext``` to obtain tokenizers for language processing:
"""
from torchtext.data.utils import get_tokenizer

"""
Initialize the German and English tokenizers using spaCy's 'de_core_news_sm' model:
"""
# Making a placeholder dict to store both tokenizers
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

"""
The line ```token_transform['de'](german)``` will tokenize the German string (or text) using the previously defined ```token_transform['de']``` for the German language.
"""
token_transform['de'](german)

"""
The same thing for English:
"""
token_transform['en'](english)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

#place holder dict for 'en' and 'de' vocab transforms
vocab_transform = {}

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    # Define a mapping to associate the source and target languages
    # with their respective positions in the data samples.
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    # Iterate over each data sample in the provided dataset iterator
    for data_sample in data_iter:
        # Tokenize the data sample corresponding to the specified language
        # and yield the resulting tokens.
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data iterator
    train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    #To decrease the number of padding tokens, you sort data on the source length to batch similar-length sequences together
    sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(sorted_dataset, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# If not set, it throws ``RuntimeError`` when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

seq_en=vocab_transform['en'](token_transform['en'](english))
print(f"English text string: {english}\n English sequence: {seq_en}")

seq_de=vocab_transform['de'](token_transform['de'](german))
print(f"German text string: {german}\n German sequence: {seq_de}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# function to add BOS/EOS, flip source sentence and create tensor for input sequence indices
def tensor_transform_s(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.flip(torch.tensor(token_ids), dims=(0,)),
                      torch.tensor([EOS_IDX])))

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform_t(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

seq_en=tensor_transform_s(seq_en)
seq_en

seq_de=tensor_transform_t(seq_de)
seq_de

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}

text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], #Tokenization
                                            vocab_transform[SRC_LANGUAGE], #Numericalization
                                            tensor_transform_s) # Add BOS/EOS and create tensor

text_transform[TGT_LANGUAGE] = sequential_transforms(token_transform[TGT_LANGUAGE], #Tokenization
                                            vocab_transform[TGT_LANGUAGE], #Numericalization
                                            tensor_transform_t) # Add BOS/EOS and create tensor

"""## Processing data in batches
The collate_fn function builds upon the utilities established earlier. It performs the text_transform to a batch of raw data. Furthermore, it ensures consistent sequence lengths within the batch through padding. This transformation readies the data for input to a transformer model designed for language translation tasks.

"""

# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
        tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
        src_batch.append(src_sequences)
        tgt_batch.append(tgt_sequences)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX,batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX,batch_first=True)

    return src_batch.to(device), tgt_batch.to(device)

BATCH_SIZE = 4

train_iterator = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_train_iterator = sorted(train_iterator, key=lambda x: len(x[0].split()))
train_dataloader = DataLoader(sorted_train_iterator, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)

valid_iterator = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
sorted_valid_dataloader = sorted(valid_iterator, key=lambda x: len(x[0].split()))
valid_dataloader = DataLoader(sorted_valid_dataloader, batch_size=BATCH_SIZE, collate_fn=collate_fn,drop_last=True)


src, trg = next(iter(train_dataloader))
src,trg

"""
© Copyright IBM Corporation. All rights reserved.

```{## Change Log}
```

```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description|}
```
```{|-|-|-|-|}
```
```{|2023-10-24|0.1|Roodra|Created Lab Template|}
```
"""
