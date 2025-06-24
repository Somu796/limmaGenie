import numpy as np
import pandas as pd
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI

### Lets start with initialisation of LLM and embeddings

### For Embeddings

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint='https://limmagenie.openai.azure.com/',# If not provided, will read env variable AZURE_OPENAI_ENDPOINT
    api_key = 'BVgOK443v5kBCMhq32znGKFmsuq1J9oSZu3MliruAtW0r11VuaasJQQJ99AKACfhMk5XJ3w3AAABACOGCOPw', # If not provided, will read env variable AZURE_OPENAI_API_KEY
    api_version = '2024-08-01-preview'
)

### For LLM

llm = ChatOpenAI(model='gpt-3.5-turbo',
    api_key = "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", )

def cosine_similarity(a, b):
    ## Function to calculate cosine similarity between two vectors
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_docs(df, user_query, top_n=4):
    ## Function to search documents based on user query
    embedding = embeddings.embed_query(user_query)
    df["similarities"] = df.embeddings.apply(lambda x: cosine_similarity(x, embedding))
    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    list_context_dict = []
    for i in range(top_n):
        list_context_dict.append({
            "Question" : res["question"].iloc[i],
            "Answer" : res["answers"].iloc[i],
            "link" : res["url"].iloc[i],
            "index" : "[i]"
        })
    final_context = ''
    for context in list_context_dict:
        final_context += f'''
{context["index"]} 
Question : {context["Question"]}
Answer : {context["Answer"]}'''
    return list_context_dict

def get_context(user_query):
    ## Function to get context based on user query
    # df = pd.read_json("limma.json")
    df = pd.read_json("limma.json")
    return search_docs(df, user_query, top_n=4)

from langchain_core.prompts import ChatPromptTemplate
prompt  = ChatPromptTemplate([
    ("system", '''
You are a Biostatistics assistant specialized in performing Limma (Linear Models for Microarray or Omics Data) analyses.
You are supposed to anser the user's query and provide the code for the same.
"Carefully perform the following instructions in order. "
"1. Retrieve relevant documents related to the user's query. "
"2. Determine which of the retrieved documents contain facts pertinent to crafting an informative response. "
"3. Construct your answer based on the information extracted from the relevant documents. Avoid directly copying any grounding markup or references, such as [1][2], from the source material. Always attribute the information by citing the corresponding document(s) using the format `[1][2]` while composing the answer. NEVER include a References or Sources section at the end of your answer. "
"4. When relevant documents are available, prioritize the information obtained from the search results over the knowledge from your pre-training data."
"Now answer user's latest query using the same language the user used: 
`CONTEXT` : {context}'''),
    ("human", '''
`User Message` : 
{user_input} 
Reply for `User Message` If code is requested, explain the code also in detail in a simple way''')
])


chain = prompt | llm

def get_response_llm(user_query, chain = chain):

    input_json = {
    "user_input": user_query,
    "context": get_context(user_query)
}

    response = chain.invoke(input_json)
    return [response.content, "n"]

# resp = generate_response(input_json, chain)
# print(resp.content)