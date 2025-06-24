from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model_name=os.environ["OPENAI_MODEL_NAME"],  # e.g., "gpt-4" or "gpt-3.5-turbo"
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
# Initialize the embeddings model
embeddings = OpenAIEmbeddings(model=os.environ["DATA_EMBEDDING_DEPLOYMENT"], openai_api_key=os.environ["OPENAI_API_KEY"])

# # Testing
# ## 🔹 Test encoding a simple sentence
# try:
#     test_text = "Hello, world!"
#     embedding_vector = embeddings.embed_query(test_text)
#     print("✅ Model is working! Embedding shape:", len(embedding_vector))
# except Exception as e:
#     print("❌ Error Embedding:", str(e))

# # ## Test the azure open ai deployment
# try:
#     response = llm.invoke("Hello! How are you?")
#     print("✅ Model is working! Response:", response)
# except Exception as e:
#     print("❌ Error Generate Msg:", str(e))
