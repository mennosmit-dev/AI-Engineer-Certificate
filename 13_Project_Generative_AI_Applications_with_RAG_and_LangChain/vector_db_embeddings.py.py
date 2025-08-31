"""
Objectives
- Prepare and preprocess documents for embeddings.
- Generate embeddings using watsonx.ai's embedding model.
- Store these embeddings in Chroma DB and FAISS.
- Perform similarity searches to retrieve relevant documents based on new inquiries.

Installing required libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install --user"ibm-watsonx-ai==1.0.4"
# !pip install  --user "langchain==0.2.1"
# !pip install  --user "langchain-ibm==0.1.7"
# !pip install  --user "langchain-community==0.2.1"
# !pip install --user "chromadb==0.4.24"
# !pip install  --user "faiss-cpu==1.8.0"

"""
Load text
"""

!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BYlUHaillwM8EUItaIytHQ/companypolicies.txt"

from langchain_community.document_loaders import TextLoader

loader = TextLoader("companypolicies.txt")
data = loader.load()

"""
You can have a look at this document.
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

chunks = text_splitter.split_documents(data)

"""
Let's take a look at how many chunks you get.
"""

len(chunks)

"""
Embedding model
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
Vector store

Chroma DB

Build the database
"""

from langchain.vectorstores import Chroma

"""
Next, you need to create an ID list that will be used to assign each chunk a unique identifier, allowing you to track them later in the vector database. The length of this list should match the length of the chunks.
"""

ids = [str(i) for i in range(0, len(chunks))]

"""
The next step is to use the embedding model to create embeddings for each chunk and then store them in the Chroma database.
"""

vectordb = Chroma.from_documents(chunks, watsonx_embedding, ids=ids)

"""
Now that you have built the vector store named `vectordb`, you can use the method `.collection.get()` to print some of the chunks indexed by their IDs.
"""

for i in range(3):
    print(vectordb._collection.get(ids=str(i)))

"""
You can also use the method `._collection.count()` to see the length of the vector database, which should be the same as the length of chunks.
"""

vectordb._collection.count()

"""
Similarity search
"""

query = "Email policy"
docs = vectordb.similarity_search(query)
docs

"""
You can specify `k = 1` to just retrieve the top one result.
"""

vectordb.similarity_search(query, k = 1)

"""
FIASS DB

Build the database
"""

from langchain_community.vectorstores import FAISS

faissdb = FAISS.from_documents(chunks, watsonx_embedding, ids=ids)

"""
Next, print the first three information pieces in the database based on IDs.
"""

for i in range(3):
    print(faissdb.docstore.search(str(i)))

"""
Similarity search
"""

query = "Email policy"
docs = faissdb.similarity_search(query)
docs

"""
Managing vector store: Adding, updating, and deleting entries
"""

text = "Instructlab is the best open source tool for fine-tuning a LLM."

from langchain_core.documents import Document

"""
Form the text into a `Document` object named `new_chunk`.
"""

new_chunk =  Document(
    page_content=text,
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)

"""
Then, the new chunk should be put into a list as the vector database only accepts documents in a list.
"""

new_chunks = [new_chunk]

"""
Before you add the document to the vector database, since there are 215 chunks with IDs from 0 to 214, if you print ID 215, the document should show no values. Let's validate it.
"""

print(vectordb._collection.get(ids=['215']))

"""
Next, you can use the method `.add_documents()` to add this `new_chunk`. In this method, you should assign an ID to the document. Since there are already IDs from 0 to 214, you can assign ID 215 to this document. The ID should be in string format and placed in a list.
"""

vectordb.add_documents(
    new_chunks,
    ids=["215"]
)

"""
Now you can count the length of the vector database again to see if it has increased by one.
"""

vectordb._collection.count()

"""
You can then print this newly added document from the database by its ID.
"""

print(vectordb._collection.get(ids=['215']))

"""
Update
"""

update_chunk =  Document(
    page_content="Instructlab is a perfect open source tool for fine-tuning a LLM.",
    metadata={
        "source": "ibm.com",
        "page": 1
    }
)

"""
Then, you can use the method `.update_document()` to update the specific stored information indexing by its ID.
"""

vectordb.update_document(
    '215',
    update_chunk,
)

print(vectordb._collection.get(ids=['215']))

""
Delete
"""

vectordb._collection.delete(ids=['215'])

print(vectordb._collection.get(ids=['215']))

"""
Exercise 1 - Use another query to conduct similarity search.

Can you use another query to conduct the similarity search?
"""
query = "Smoking policy"
docs = vectordb.similarity_search(query)
docs

### Other Contributors

[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)

Joseph has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

```{## Change Log}
```

```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-24|0.1|Kang Wang|Create the lab|}
```

Copyright © IBM Corporation. All rights reserved.
"""
