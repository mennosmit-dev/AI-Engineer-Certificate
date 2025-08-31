"""Objectives
 - Use commonly used text splitters from LangChain.
 - Split source documents into chunks for downstream use in RAG

Installing required libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install "langchain==0.2.7"
# !pip install "langchain-core==0.2.20"
# !pip install "langchain-text-splitters==0.2.2"
# !pip install "lxml==5.2.2"

"""
A long document has been prepared for this project to demonstrate the performance of each splitter. Run the following code to download it.
"""

!wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YRYau14UJyh0DdiLDdzFcA/companypolicies.txt"

"""
Let's take a look at what the document looks like.
"""

# This is a long document you can split up.
with open("companypolicies.txt") as f:
    companypolicies = f.read()

print(companypolicies)

"""
Document object
"""

from langchain_core.documents import Document
Document(page_content="""Python is an interpreted high-level general-purpose programming language.
                        Python's design philosophy emphasizes code readability with its notable use of significant indentation.""",
         metadata={
             'my_document_id' : 234234,
             'my_document_source' : "About Python",
             'my_document_create_time' : 1680013019
         })

"""
Split by Character
"""

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

"""
You will use `split_text` function to operate the split.
"""

texts = text_splitter.split_text(companypolicies)

"""
Let's take a look how the document has been split.
"""

texts

"""
After the split, you'll see that the document has been divided into multiple chunks, with some character overlaps between the chunks.
"""

len(texts)

"""
You get `87` chunks.
"""

texts = text_splitter.create_documents([companypolicies], metadatas=[{"document":"Company Policies"}])  # pass the metadata as well

texts[0]

"""
Recursively Split by Character

"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

"""
Here, you are using the same document "companypolicies.txt" from earlier as an example to show the performance of `RecursiveCharacterTextSplitter`.
"""

texts = text_splitter.create_documents([companypolicies])

texts

"""
From the split results, you can see that the splitter uses recursion as the core strategy to divide the document into chunks.
"""

len(texts)

"""
Split Code
"""

from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

"""Use the following to see a list of codes it supports.

"""

[e.value for e in Language]

"""
Use the following code to see what default separators it uses, for example, for Python.
"""

RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON)

"""
Python
"""

PYTHON_CODE = """
    def hello_world():
        print("Hello, World!")

    # Call the function
    hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
python_docs = python_splitter.create_documents([PYTHON_CODE])
python_docs

"""
Javascript
"""

RecursiveCharacterTextSplitter.get_separators_for_language(Language.JS)

"""
The following code is used to separate the JSON language code.
"""

JS_CODE = """
    function helloWorld() {
      console.log("Hello, World!");
    }

    // Call the function
    helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)
js_docs = js_splitter.create_documents([JS_CODE])
js_docs

"""
Markdown Header Text Splitter
"""

from langchain.text_splitter import MarkdownHeaderTextSplitter

"""
For example, if you want to split this markdown:
"""

md = "# Foo\n\n## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n### Boo \n\nHi this is Lance \n\n## Baz\n\nHi this is Molly"

"""
You can specify the headers to split on:
"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(md)
md_header_splits

"""
From the split results, you can see that the Markdown file is divided into several chunks formatted as document objects. The `page_content` contains the text under the headings, and the `metadata` contains the header information corresponding to the `page_content`.
"""

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
md_header_splits = markdown_splitter.split_text(md)
md_header_splits

"""
Split by HTML

Split by HTML header
"""

from langchain_text_splitters import HTMLHeaderTextSplitter

"""
Assume you have the following HTML code that you want to split.
"""

html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
"""

"""
Set up the header to split.
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

"""
Split the HTML string using `HTMLHeaderTextSplitter`.
"""

html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits

"""
Split by HTML section
"""

from langchain_text_splitters import HTMLSectionSplitter

html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>Foo</h1>
            <p>Some intro text about Foo.</p>
            <div>
                <h2>Bar main section</h2>
                <p>Some intro text about Bar.</p>
                <h3>Bar subsection 1</h3>
                <p>Some text about the first subtopic of Bar.</p>
                <h3>Bar subsection 2</h3>
                <p>Some text about the second subtopic of Bar.</p>
            </div>
            <div>
                <h2>Baz</h2>
                <p>Some text about Baz</p>
            </div>
            <br>
            <p>Some concluding text about Foo</p>
        </div>
    </body>
    </html>
"""

headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]

html_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)
html_header_splits = html_splitter.split_text(html_string)
html_header_splits

"""
Exercise 1 - Changing separator for CharacterTextSplitter
Try to change to use another separator, for example `"\n"` to see how it affect the split and chunks.
"""

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)
texts = text_splitter.split_text(companypolicies)
texts

"""
### Exercise 2 - Splitting Latex code

Here is an example of Latex code. Try to split it.
"""

latex_text = """
    \documentclass{article}

    \begin{document}

    \maketitle

    \section{Introduction}
    Large language models (LLMs) are a type of machine learning model that can be trained on vast amounts of text data to generate human-like language. In recent years, LLMs have made significant advances in a variety of natural language processing tasks, including language translation, text generation, and sentiment analysis.

    \subsection{History of LLMs}
    The earliest LLMs were developed in the 1980s and 1990s, but they were limited by the amount of data that could be processed and the computational power available at the time. In the past decade, however, advances in hardware and software have made it possible to train LLMs on massive datasets, leading to significant improvements in performance.

    \subsection{Applications of LLMs}
    LLMs have many applications in industry, including chatbots, content creation, and virtual assistants. They can also be used in academia for research in linguistics, psychology, and computational linguistics.

    \end{document}
"""

latex_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.LATEX, chunk_size=60, chunk_overlap=0
)
latex_docs = latex_splitter.create_documents([latex_text])
latex_docs

"""
### Other Contributors

[Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)

Joseph has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.

```{## Change Log}
```

```{Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2024-07-16|0.1|Kang Wang|Create the lab|}
```

© Copyright IBM Corporation. All rights reserved.
"""
