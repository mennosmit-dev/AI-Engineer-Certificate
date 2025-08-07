"""
This project introduces language modeling by focusing on generating 90s rap lyrics. You will implement histogram N-gram models using the Natural Language Toolkit (NLTK) to analyze and understand word frequencies and distributions. This approach helps reveal linguistic patterns and the rhythmic flow characteristic of 90s rap.

# Objectives
 - Utilize histogram N-gram models, implemented through the Natural Language Toolkit (NLTK), to analyze and understand word frequencies and distributions.
"""

"""
Remove all non-word characters (everything except numbers and letters)
"""
def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

"""
Let's consider the following song lyrics to determine if you can generate similar output using a given word.
"""

song= """We are no strangers to love
You know the rules and so do I
A full commitments what Im thinking of
You wouldnt get this from any other guy
I just wanna tell you how Im feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Weve known each other for so long
Your hearts been aching but youre too shy to say it
Inside we both know whats been going on
We know the game and were gonna play it
And if you ask me how Im feeling
Dont tell me youre too blind to see
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Weve known each other for so long
Your hearts been aching but youre too shy to say it
Inside we both know whats been going on
We know the game and were gonna play it
I just wanna tell you how Im feeling
Gotta make you understand
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you
Never gonna give you up
Never gonna let you down
Never gonna run around and desert you
Never gonna make you cry
Never gonna say goodbye
Never gonna tell a lie and hurt you"""

"""
NLTK is indeed a widely-used open-source library in Python that is specifically designed for various natural language processing (NLP) tasks. It provides a comprehensive set of tools, resources, and algorithms that aid in the analysis and manipulation of human language data.
To achieve the goal, you will utilize the```word_tokenize```function. During this process, you will remove punctuation, symbols, and capital letters.
"""
from nltk.tokenize import word_tokenize
def preprocess(words):
    tokens=word_tokenize(words)
    tokens=[preprocess_string(w)  for w in tokens]
    return [w.lower()  for w in tokens if len(w)!=0 or not(w in string.punctuation) ]

tokens=preprocess(song)

"""
The outcome is a collection of tokens, wherein each element of the```tokens```pertains to the lyrics of the song, arranged in sequential order.
"""
tokens[0:10]

"""The frequency distribution of words in a sentence represents how often each word appears in that particular sentence. It provides a count of the occurrences of individual words, allowing you to understand which words are more common or frequent within the given sentence. Let's work with the following toy example:
"""
# Create a frequency distribution of words
fdist = nltk.FreqDist(tokens)
fdist

""" 
Plot the words with the top ten frequencies.
"""
plt.bar(list(fdist.keys())[0:10],list(fdist.values())[0:10])
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.show()

"""### Unigram model
An unigram model is a simple type of language model that considers each word in a sequence independently, without taking into account the previous words. In other words, it models the probability of each word occurring in the text, regardless of what came before it. Unigram models can be seen as a special case of n-gram models, where n is 1.
"""
#total count of each word
C=sum(fdist.values())
C

"""
Find the probability of the word wish  i.w $P(strangers)$.
"""
fdist['strangers']/C

"""
Also, find each individual word by converting the tokens to a set.
"""
vocabulary=set(tokens)

"""
### Bigram model

Bigrams represent pairs of consecutive words in the given phrase, i.e., $(w_{t-1},w_t)$. Consider the following words from your example: "I like dogs and I kinda like cats."
"""
bigrams = nltk.bigrams(tokens)
bigrams

"""
Convert a generator into a list, where each element of the list is a bigram.
"""
my_bigrams=list(nltk.bigrams(tokens))

"""
You can see the first 10 bigrams.
"""
my_bigrams[0:10]

"""
Compute the frequency distribution of the bigram $C(w_{t},w_{t-1})$ using the NLTK function```bigrams```.
"""
freq_bigrams  = nltk.FreqDist(nltk.bigrams(tokens))
freq_bigrams

"""
The result is akin to a dictionary, where the key is a tuple containing the bigram.
"""
freq_bigrams[('we', 'are')]

"""
It is possible to provide you with the first 10 values of the frequency distribution.
"""
for my_bigram in  my_bigrams[0:10]:
    print(my_bigram)
    print(freq_bigrams[my_bigram])

"""
Here, you can generate the conditional distribution by normalizing the frequency distribution of unigrams. In this case, you are doing it for the word 'strangers' and then sorting the results:
"""
word="strangers"
vocab_probabilities={}
for next_word in vocabulary:
    vocab_probabilities[next_word]=freq_bigrams[(word,next_word)]/fdist[word]
vocab_probabilities=sorted(vocab_probabilities.items(), key=lambda x:x[1],reverse=True)

"""
Print out the words that are more likely to occur.
"""
vocab_probabilities[0:4]

"""
Create a function to calculate the conditional probability of $W_t$ given $W_{t-1}$, sort the results, and output them as a list.
"""
def make_predictions(my_words, freq_grams, normlize=1, vocabulary=vocabulary):
    """
    Generate predictions for the conditional probability of the next word given a sequence.

    Args:
        my_words (list): A list of words in the input sequence.
        freq_grams (dict): A dictionary containing frequency of n-grams.
        normlize (int): A normalization factor for calculating probabilities.
        vocabulary (list): A list of words in the vocabulary.

    Returns:
        list: A list of predicted words along with their probabilities, sorted in descending order.
    """

    vocab_probabilities = {}  # Initialize a dictionary to store predicted word probabilities

    context_size = len(list(freq_grams.keys())[0])  # Determine the context size from n-grams keys

    # Preprocess input words and take only the relevant context words
    my_tokens = preprocess(my_words)[0:context_size - 1]

    # Calculate probabilities for each word in the vocabulary given the context
    for next_word in vocabulary:
        temp = my_tokens.copy()
        temp.append(next_word)  # Add the next word to the context

        # Calculate the conditional probability using the frequency information
        if normlize!=0:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)] / normlize
        else:
            vocab_probabilities[next_word] = freq_grams[tuple(temp)]
    # Sort the predicted words based on their probabilities in descending order
    vocab_probabilities = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)

    return vocab_probabilities  # Return the sorted list of predicted words and their probabilities

"""
Set $W_{t-1}$ to 'i' and then calculate all the values of $P(W_t | W_{t-1}=i)$.
"""
my_words="are"
vocab_probabilities=make_predictions(my_words,freq_bigrams,normlize=fdist['i'])
vocab_probabilities[0:10]

"""
The word with the highest probability, denoted as $\hat{W}_t$, is given by the first element of the list, this can be used as a simple autocomplete:
"""
vocab_probabilities[0][0]

"""
Generate a sequence using the bigram model by leveraging the preceding word (t-1) to predict and generate the subsequent word in the sequence.
"""
my_song=""
for w in tokens[0:100]:
  my_word=make_predictions(w,freq_bigrams)[0][0]
  my_song+=" "+my_word
my_song

"""
Create a sequence using the n-gram model by initiating the process with the first word in the sequence and producing an initial output. Subsequently, utilize this output as the basis for generating the next word in the sequence, i.e., you will give your model a word, then use the output to predict the next word and repeat.
"""

my_song="i"

for i in range(100):
    my_word=make_predictions(my_word,freq_bigrams)[0][0]
    my_song+=" "+my_word

my_song

"""
This method may not yield optimal results;

Trigram models incorporate conditional probability as well. The probability of a word depends on the two preceding words. The conditional probability $P(W_t | W_{t-2}, W_{t-1})$ is utilized to predict the likelihood of word $W_t$ following the two previous words in a sequence. The context is $W_{t-2}, W_{t-1}$ and is of length 2. Let's compute the conditional probability for each trigram:
"""

freq_trigrams  = nltk.FreqDist(nltk.trigrams(tokens))
freq_trigrams

"""
Find the probability for each of the next words.
"""
make_predictions("so do",freq_trigrams,normlize=freq_bigrams[('do','i')] )[0:10]

"""
Find the probability for each of the next words.
"""
my_song=""
w1=tokens[0]
for w2 in tokens[0:100]:
    gram=w1+' '+w2
    my_word=make_predictions(gram,freq_trigrams )[0][0]
    my_song+=" "+my_word
    w1=w2

my_song

"""
There are various challenges associated with Histogram-Based Methods, some of which are quite straightforward. For instance, when considering the case of having N words in your vocabulary, a Unigram model would entail $N$ bins, while a Bigram model would result in $N^2$ bins and so forth.
N-gram models also encounter limitations in terms of contextual understanding and their ability to capture intricate word relationships. For instance, let's consider the phrases `I hate dogs`, `I don’t like dogs`, and **don’t like** means **dislike**. Within this context, a histogram-based approach would fail to grasp the significance of the phrase **don’t like** means **dislike**, thereby missing out on the essential semantic relationship it encapsulates.
---

# Congratulations! You have completed the lab

## Authors

[Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/) has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

### Contributor

[Roodra Kanwar](https://www.linkedin.com/in/roodrakanwar/) is completing his MS in CS specializing in big data from Simon Fraser University. He has previous experience working with machine learning and as a data engineer.

```{## Change log}

```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2023-09-01|0.1|Joseph|Created Lab Template & Guided Project||2023-09-03|0.1|Joseph|Updated Guided Project|}

© Copyright IBM Corporation. All rights reserved.
"""
