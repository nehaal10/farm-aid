from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_community.llms.openai import OpenAI
import os


os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# load the pdf
file = "data/farmerbook.pdf"
loader = UnstructuredPDFLoader(file)
documents = loader.load()
# i am prints the no document and characters in document
print(f'You have {len(documents)} document(s) in your data')
print(f'There are {len(documents[0].page_content)} characters in your document')

# The pdf is splitted in small chucks of documents
spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
data = spliter.split_documents(documents)

print(f'Now you have {len(data)} documents')


embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Create pf pinecone once alone run it .
# index_name="langchain1"
# pc.create_index(
#     name='langchain3', 
#     dimension=1536, 
#     metric="cosine", 
#     spec=PodSpec(environment="gcp-starter")
# )
index_name="langchain3"
docsearch = PineconeVectorStore.from_documents(data, embeddings, index_name=index_name)

query = "What did the president say about Ketanji Brown Jackson"
docs = docsearch.similarity_search(query)
print(docs)



# vectorestore = Chroma.from_documents(data,embeddings)

# llm = OpenAI(temperature = 0 , open_api_key=OPENAI_API_KEY)
# chain = load_qa_chain(llm, chain_type="stuff")

# query = "whats the ideal distance between mango trees"
# docs = vectorestore.similarity_search(query)

# chain.run(input_documents=docs, question=query)
