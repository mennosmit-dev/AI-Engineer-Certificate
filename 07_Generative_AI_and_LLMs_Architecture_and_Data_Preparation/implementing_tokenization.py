"""
Applying several tokenisation strategies including NLTK, SPACY, BERT (Unigram and SentencePiece) and XLNetTokenizer (and torchtext).

# Objectives
 - Understand the concept of tokenization and its importance in natural language processing
 - Identify and explain word-based, character-based, and subword-based tokenization methods.
 - Apply tokenization strategies to preprocess raw textual data before using it in machine learning models.
"""
!pip install nltk
!pip install transformers==4.42.1
!pip install sentencepiece
!pip install spacy
!python -m spacy download en_core_web_sm
!python -m spacy download de_core_news_sm
!pip install scikit-learn
!pip install torch==2.2.2
!pip install torchtext==0.17.2
!pip install numpy==1.26.0

"""
### Importing required libraries
"""
import nltk
nltk.download("punkt")
nltk.download('punkt_tab')
import spacy
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
from transformers import BertTokenizer
from transformers import XLNetTokenizer
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

text = "This is a sample sentence for word tokenization."
tokens = word_tokenize(text)
print(tokens)

"""
General libraries like nltk and spaCy often split words like 'don't' and 'couldn't,' which are contractions, into different individual words. There's no universal rule, and each library has its own tokenization rules for word-based tokenizers. However, the general guideline is to preserve the input format after tokenization to match how the model was trained.
"""
# This showcases word_tokenize from nltk library
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
tokens = word_tokenize(text)
print(tokens)

# This showcases the use of the 'spaCy' tokenizer with torchtext's get_tokenizer function
text = "I couldn't help the dog. Can't you do it? Don't be afraid if you are."
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
# Making a list of the tokens and priting the list
token_list = [token.text for token in doc]
print("Tokens:", token_list)
# Showing token details
for token in doc:
    print(token.text, token.pos_, token.dep_)

text = "Unicorns are real. I saw a unicorn yesterday."
token = word_tokenize(text)
print(token)

"""
## Character-based tokenizer

## Subword-based tokenizer

### WordPiece

Now, the WordPiece tokenizer is implemented in BertTokenizer.
Note that BertTokenizer treats composite words as separate tokens.
"""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("IBM taught me tokenization.")

"""
### Unigram and SentencePiece
"""
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("IBM taught me tokenization.")

"""
## Tokenization with PyTorch
In PyTorch, especially with the `torchtext` library, the tokenizer breaks down text from a data set into individual words or subwords, facilitating their conversion into numerical format. After tokenization, the vocab (vocabulary) maps these tokens to unique integers, allowing them to be fed into neural networks. This process is vital because deep learning models operate on numerical data and cannot process raw text directly. Thus, tokenization and vocabulary mapping serve as a bridge between human-readable text and machine-operable numerical data. Consider the dataset:
"""
dataset = [
    (1,"Introduction to NLP"),
    (2,"Basics of PyTorch"),
    (1,"NLP Techniques for Text Classification"),
    (3,"Named Entity Recognition with PyTorch"),
    (3,"Sentiment Analysis using PyTorch"),
    (3,"Machine Translation with PyTorch"),
    (1," NLP Named Entity,Sentiment Analysis,Machine Translation "),
    (1," Machine Translation with NLP "),
    (1," Named Entity vs Sentiment Analysis  NLP ")]

"""
This next line imports the ```get_tokenizer``` function from the ```torchtext.data.utils``` module. In the torchtext library, the ```get_tokenizer```  function is utilized to fetch a tokenizer by name. It provides support for a range of tokenization methods, including basic string splitting, and returns various tokenizers based on the argument passed to it.
"""
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer("basic_english")

"""
You apply the tokenizer to the dataset. 
"""
tokenizer(dataset[0][1])

"""
## Token indices
"""
def yield_tokens(data_iter):
    for  _,text in data_iter:
        yield tokenizer(text)

my_iterator = yield_tokens(dataset)

"""
This creates an iterator called **```my_iterator```** using the generator. To begin the evaluation of the generator and retrieve the values, you can iterate over **```my_iterator```** using a for loop or retrieve values from it using the **```next()```** function.
"""
next(my_iterator)

"""
### Out-of-vocabulary (OOV)
"""
vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

"""
This code demonstrates how to fetch a tokenized sentence from an iterator, convert its tokens into indices using a provided vocabulary, and then print both the original sentence and its corresponding indices.
"""
def get_tokenized_sentence_and_indices(iterator):
    tokenized_sentence = next(iterator)  # Get the next tokenized sentence
    token_indices = [vocab[token] for token in tokenized_sentence]  # Get token indices
    return tokenized_sentence, token_indices
tokenized_sentence, token_indices = get_tokenized_sentence_and_indices(my_iterator)
next(my_iterator)
print("Tokenized Sentence:", tokenized_sentence)
print("Token Indices:", token_indices)

"""
Using the lines of code provided above in a simple example, demonstrate tokenization and the building of vocabulary in PyTorch.
"""
lines = ["IBM taught me tokenization",
         "Special tokenizers are ready and they will blow your mind",
         "just saying hi!"]
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
tokens = []
max_length = 0
for line in lines:
    tokenized_line = tokenizer_en(line)
    tokenized_line = ['<bos>'] + tokenized_line + ['<eos>']
    tokens.append(tokenized_line)
    max_length = max(max_length, len(tokenized_line))
for i in range(len(tokens)):
    tokens[i] = tokens[i] + ['<pad>'] * (max_length - len(tokens[i]))
print("Lines after adding special tokens:\n", tokens)
# Build vocabulary without unk_init
vocab = build_vocab_from_iterator(tokens, specials=['<unk>'])
vocab.set_default_index(vocab["<unk>"])
# Vocabulary and Token Ids
print("Vocabulary:", vocab.get_itos())
print("Token IDs for 'tokenization':", vocab.get_stoi())

new_line = "I learned about embeddings and attention mechanisms."
# Tokenize the new line
tokenized_new_line = tokenizer_en(new_line)
tokenized_new_line = ['<bos>'] + tokenized_new_line + ['<eos>']
# Pad the new line to match the maximum length of previous lines
new_line_padded = tokenized_new_line + ['<pad>'] * (max_length - len(tokenized_new_line))
# Convert tokens to IDs and handle unknown words
new_line_ids = [vocab[token] if token in vocab else vocab['<unk>'] for token in new_line_padded]
# Example usage
print("Token IDs for new line:", new_line_ids)

text = """
Going through the world of tokenization has been like walking through a huge maze made of words, symbols, and meanings. Each turn shows a bit more about the cool ways computers learn to understand our language. And while I'm still finding my way through it, the journey’s been enlightening and, honestly, a bunch of fun.
Eager to see where this learning path takes me next!"
"""
# Counting and displaying tokens and their frequency
from collections import Counter
def show_frequencies(tokens, method_name):
    print(f"{method_name} Token Frequencies: {dict(Counter(tokens))}\n")
    
import time
# First I am trying NlTK.
start = time.time()
tokens = word_tokenize(text)
end = time.time()
print(f"NLTK number of tokens is {len(tokens)}, and time is {end-start}")

# Secondly I am trying Spacy.
nlp = spacy.load("en_core_web_sm")
start = time.time()
doc = nlp(text)
token_list = [token.text for token in doc]
end = time.time()
print(f"Spacy number of tokens is {len(tokens)}, and time is {end-start}")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
start = time.time()
tokens = tokenizer.tokenize(text)
end = time.time()
print(f"BERT number of tokens is {len(tokens)}, and time is {end-start}")

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
start = time.time()
tokens = tokenizer.tokenize(text)
end = time.time()
print(f"XLNetTokenizer number of tokens is {len(tokens)}, and time is {end-start}")

# Congratulations! You have completed the lab

## Authors

[Roodra Kanwar](https://www.linkedin.com/in/roodrakanwar/) is completing his MS in CS specializing in big data from Simon Fraser University. He has previous experience working with machine learning and as a data engineer.

[Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/) has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

[Vicky Kuo](https://www.linkedin.com/in/vicky-tck/) is completing her Master's degree in IT at York University with scholarships. Her master's thesis explores the optimization of deep learning algorithms, employing an innovative approach to scrutinize and enhance neural network structures and performance.

© Copyright IBM Corporation. All rights reserved.

```{## Change Log}
```

```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description|}
```
```{|-|-|-|-|}
```
```{|2023-10-02|0.1|Roodra|Created Lab Template|}
```
```{|2023-10-03|0.1|Vicky|Revised the Lab|}
```
"""
