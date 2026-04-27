from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
def run_rag():
   # Load PDF
   loader = PyPDFLoader("sample.pdf")
   documents = loader.load()
   # Split text
   splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
   texts = splitter.split_documents(documents)
   # Create embeddings
   embeddings = OpenAIEmbeddings()
   # Store in vector DB
   db = FAISS.from_documents(texts, embeddings)
   # Create QA chain
   qa = RetrievalQA.from_chain_type(
       llm=ChatOpenAI(),
       retriever=db.as_retriever()
   )
   # Ask question
   query = input("Ask a question: ")
   result = qa.run(query)
   print("\nAnswer:", result)
if __name__ == "__main__":
   run_rag()
