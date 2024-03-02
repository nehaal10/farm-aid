from langchain_community.llms import llamacpp
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import vector_stores, SimpleDirectoryReader, ServiceContext
from llama_index.prompts.prompts import SimpleInputPrompt

llm = llamacpp(
    model_path = "model/llama-2-7b-chat.ggmlv3.q8_0.bin", verbose = True
)



documents = SimpleDirectoryReader("./documents").load_data()


system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the question and context provided."
query_wrapper_prompt = SimpleInputPrompt("<|USER|>{query_str}<|ASSISTANT|>")

embeded_model = LangchainEmbedding(
  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeded_model
)

index = vector_stores.FaissVectorStore.from_documents(documents, service_context=service_context)

engine = index.as_query_engine()
response = engine.query("what should be the distance from one mongo tree to another")

print(response)



