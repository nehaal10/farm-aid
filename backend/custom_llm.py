from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# insert open ai key

file = "data/farmerbook.pdf"
loader = UnstructuredFileLoader(file)
documents = loader.load()

spliter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
data = spliter.split_documents(documents)

# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vectorestore = Chroma.from_documents(data,embeddings)

# llm = OpenAI(temperature = 0 , open_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

query = "whats the ideal distance between mango trees"
docs = vectorestore.similarity_search(query)

chain.run(input_documents=docs, question=query)
