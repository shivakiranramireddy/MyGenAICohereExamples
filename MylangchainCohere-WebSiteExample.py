## RAG EXAMPLE -> Url Loader -> Oracle url

import os 

os.environ["COHERE_API_KEY"] = "<Your API KEY here>"

from langchain_community.chat_models import ChatCohere

llm = ChatCohere()

# pip install beautifulsoup4

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://docs.oracle.com/en/cloud/paas/autonomous-database/serverless/adbsb/dbms-cloud-subprograms.html")

docs = loader.load()

from langchain_community.embeddings import CohereEmbeddings

embeddings = CohereEmbeddings()

# pip install faiss-cpu --> local vector store 

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

from langchain.chains import create_retrieval_chain

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "how can we use DBMS_CLOUD package to create credentials?"})
print(response["answer"])

# LangSmith offers several features that can help with testing:...