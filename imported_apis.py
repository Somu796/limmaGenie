from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI, ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os

from openai import embeddings

load_dotenv()

# print(os.environ["AZURE_OPENAI_API_KEY"])

# get model from Azure
# llm = AzureChatOpenAI(
#     deployment_name=os.environ["LLM_RESPONSE_DEPLOYMENT"],  # Replace this with your azure deployment name
#     api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_version=os.environ["LLM_RESPONSE_AZURE_OPENAI_API_VERSION"],
# )

# # You need to deploy your own embedding model as well as your own chat completion model
# embeddings = AzureOpenAIEmbeddings(
#     azure_deployment=os.environ["DATA_EMBEDDING_DEPLOYMENT"],
#     api_key=os.environ["AZURE_OPENAI_API_KEY"],
#     azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
#     api_version=os.environ["DATA_EMBEDDING_AZURE_OPENAI_API_VERSION"],
# )

# embeddings 


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



# # # Check firewall status
# # def check_firewall():
# #     test_url = "https://api.openai.com/v1/models"  # OpenAI API
# #     try:
# #         response = requests.get(test_url, verify=False)  # Disable SSL verification
# #         if response.status_code == 200:
# #             print("✅ Firewall check passed. API is reachable.")
# #             return True
# #         elif response.status_code == 401:
# #             print("❌ Firewall check failed. Unauthorized (401). Check your API key.")
# #             if not os.getenv("LLM_RESPONSE_OPEN_AI_KEY"):
# #                 print("❌ API key for LLM_RESPONSE_OPEN_AI_KEY is not set. Please set it in your .env file.")
# #             return False
# #         else:
# #             print(f"❌ Firewall check failed. Status code: {response.status_code}")
# #             return False
# #     except requests.exceptions.RequestException as e:
# #         print(f"❌ Firewall check failed. Error: {str(e)}")
# #         return False

# # if check_firewall():
# #     # Test the model with a simple prompt
# #     max_retries = 3
# #     retry_delay = 5  # seconds

# #     for attempt in range(max_retries):
# #         try:
# #             response = llm.invoke("Hello! How are you today?")
# #             print("✅ Model is working! Response:", response)
# #             break
# #         except Exception as e:
# #             print(f"❌ Error on attempt {attempt + 1}: {str(e)}")
# #             if attempt < max_retries - 1:
# #                 print(f"Retrying in {retry_delay} seconds...")
# #                 time.sleep(retry_delay)
# #             else:
# #                 print("❌ Failed after multiple attempts.")
# # else:
# #     print("❌ Firewall is blocking the connection. Please check your network settings.")
