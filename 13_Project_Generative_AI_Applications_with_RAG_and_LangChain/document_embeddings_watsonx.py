"""
Objectives
 - Prepare and preprocess documents for embedding
 - Use watsonx.ai and Hugging Face embedding models to generate embeddings for your documents

Installing required libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# #After executing the cell,please RESTART the kernel and run all the cells.
# !pip install --user "ibm-watsonx-ai==1.1.2"
# !pip install --user "langchain==0.2.11"
# !pip install --user "langchain-ibm==0.1.11"
# !pip install --user "langchain-community==0.2.10"
# !pip install --user "sentence-transformers==3.0.1"

"""
Load data
"""

!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/i5V3ACEyz6hnYpVq6MTSvg/state-of-the-union.txt"

from langchain_community.document_loaders import TextLoader

loader = TextLoader("state-of-the-union.txt")
data = loader.load()

"""
Let's take a look at the document.
"""

data

"""
Split data
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_text(data[0].page_content)

"""
Let's see how many chunks you get.
"""

len(chunks)

"""
Let's also see what these chunks looks like.
"""

chunks

"""
Watsonx embedding model

Model description

Build model
"""

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

"""
Query embeddings
"""

query = "How are you?"

query_result = watsonx_embedding.embed_query(query)

"""
Let's see the length/dimension of this embedding.
"""

len(query_result)

"""
It has a dimension of `768`, which aligns with the model description.
"""

query_result[:5]

"""
Document embeddings
"""

doc_result = watsonx_embedding.embed_documents(chunks)

"""
As each piece of text is embedded into a vector, so the length of the `doc_result` should be the same as the length of chunks.
"""

len(doc_result)

"""
Now, take a look at the first five results from the embeddings of the first piece of text.
"""

doc_result[0][:5]

"""
Check the embedding dimension to see if it is also 768.
"""

len(doc_result[0])

"""
Hugging Face embedding model

Model description

Build model
"""

from langchain_community.embeddings import HuggingFaceEmbeddings

"""
Then, you specify the model name.
"""

model_name = "sentence-transformers/all-mpnet-base-v2"

"""
Here we create a embedding model object.
"""

huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)

"""
Query embeddings
"""

query = "How are you?"

query_result = huggingface_embedding.embed_query(query)

query_result[:5]

"""
Document embeddings
"""

doc_result = huggingface_embedding.embed_documents(chunks)
doc_result[0][:5]

len(doc_result[0])

"""
### Exercise 1 - Using another watsonx embedding model
Watsonx.ai also supports other embedding models, for which you can find more information [here](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-slate-30m-english-rtrvr-model-card.html?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Embed+documents+with+watsonx%E2%80%99s+embedding_v1_1721662184&context=wx). Can you try to use another embedding model to create embeddings for the document?
"""

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}

watsonx_embedding = WatsonxEmbeddings(
    model_id="ibm/slate-30m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params,
)

doc_result = watsonx_embedding.embed_documents(chunks)

doc_result[0][:5]

"""

### Other Contributors

[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)

Joseph has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

[Hailey Quach](https://author.skills.network/instructors/hailey_quach)

Hailey is a Data Scientist Intern at IBM. She is also pursuing a BSc in Computer Science, Honors at Concordia University, Montreal.

```{## Change Log}
```

```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-22|0.1|Kang Wang|Create the lab|}
```

Copyright © IBM Corporation. All rights reserved.
"""
