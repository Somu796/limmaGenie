from pymongo import MongoClient
from pymongo import errors
from search import searchQuery
from imported_apis import llm, embeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from answer_reranker import calculate_cosine_similarity, filter_and_rank_results
import numpy as np
import os
import time
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.getenv("LLM_RESPONSE_DEPLOYMENT", "Not Found")
# Parameters
CONFIG = {
    "DB_NAME": "limmaGenie_Database",
    "COLLECTION_NAME": "merged_data",
    "INDEX_NAME": "merged_data_index",
    "NO_MATCH_RETURN": 5,
}

class AtlasClient:
    def __init__(self, atlas_uri, dbname):
        """
        Initialize MongoDB client with error handling.
        
        :param atlas_uri: Connection string for MongoDB
        :param dbname: Name of the database
        """
        try:
            self.mongodb_client = MongoClient(atlas_uri)
            self.database = self.mongodb_client[dbname]
            logger.info(f"Connected to database: {dbname}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def ping(self):
        """
        Quick method to test Atlas instance connection.
        
        :raises: ConnectionError if ping fails
        """
        try:
            self.mongodb_client.admin.command('ping')
            logger.info("Successfully pinged the database")
        except Exception as e:
            logger.error(f"Ping failed: {e}")
            raise

    def get_collection(self, collection_name):
        """
        Retrieve a specific collection.
        
        :param collection_name: Name of the collection
        :return: MongoDB collection object
        """
        return self.database[collection_name]

    def find(self, collection_name, filter=None, limit=10):
        """
        Find documents in a collection.
        
        :param collection_name: Name of the collection
        :param filter: Query filter
        :param limit: Maximum number of documents to return
        :return: List of documents
        """
        filter = filter or {}
        collection = self.database[collection_name]
        return list(collection.find(filter=filter, limit=limit))

    def vector_search(self, collection_name, index_name, attr_name, embedding_vector, limit=5):
        """
        Perform vector search on a collection.
        
        :param collection_name: Name of the collection
        :param index_name: Name of the vector search index
        :param attr_name: Attribute to search on
        :param embedding_vector: Query embedding vector
        :param limit: Maximum number of results
        :return: List of search results
        """
        collection = self.database[collection_name]
        try:
            results = collection.aggregate([
                {
                    '$vectorSearch': {
                        "index": index_name,
                        "path": attr_name,
                        "queryVector": embedding_vector,
                        "numCandidates": 50,
                        "limit": limit,
                    }
                },
                {
                    "$project": {
                        '_id': 1,
                        'url': 1,
                        'title': 1,
                        'question': 1,
                        'answers': 1,
                        'code': 1,
                        "search_score": {"$meta": "vectorSearchScore"}
                    }
                }
            ])
            return list(results)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def close_connection(self):
        """
        Close the MongoDB client connection.
        """
        try:
            self.mongodb_client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")


def process_vector_search_results(
    query: str, 
    embeddings_func, 
    config: dict,
    seed: int = 42  # Add deterministic seed
):
    """
    Perform vector search with improved relevance filtering.
    
    :param query: Search query string
    :param embeddings_func: Function to generate embeddings
    :param config: Configuration dictionary
    :param seed: Random seed for reproducibility
    :return: Dictionary with processed search results
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Clean up query
    query = query.lower().strip()
    
    # Generate embeddings
    try:
        query_embedding = embeddings_func.embed_query(query)
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return {
            "question": query,
            "content": "",
            "urls": "",
            "status": f"Embedding Error: {e}"
        }

    # Perform vector search
    atlas_client = None
    try:
        atlas_client = AtlasClient(
            os.environ["MONGODB_CONNECTION_STRING"], 
            config["DB_NAME"]
        )
        
        # Retrieve documents
        answers = atlas_client.vector_search(
            collection_name=config["COLLECTION_NAME"],
            index_name=config["INDEX_NAME"],
            attr_name='embedding',
            embedding_vector=query_embedding,
            limit=config.get("NO_MATCH_RETURN", 10)
        )
        
        # Filter and rank results
        filtered_answers = filter_and_rank_results(
            answers, 
            query_embedding, 
            top_k=3  # Limit to top 3 most relevant results
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return {
            "question": query,
            "content": "",
            "urls": "",
            "status": f"Search Error: {e}"
        }
    finally:
        if atlas_client:
            atlas_client.close_connection()

    # Process results with improved citation support
    content_with_citations = []
    reference_map = {}

    # Track URLs already added
    # seen_urls = set()
    
    for idx, answer in enumerate(filtered_answers, 1):
        # Create a unique reference entry
        url = answer.get('url', 'N/A')
        
        if url != 'N/A':
            # seen_urls.add(url)
            if ',' in url:
                # Split URL if it contains multiple parts
                url = url.split(",")[0] + "  " + url.split(",")[1]
        reference_map[idx] = url

        
        # Process answers with citations
        if answer.get('answers'):
            for answer_text in answer['answers']:
                cited_answer = {
                    'text': answer_text.strip().replace(chr(10), ''),
                    'citation': f'[{idx}]',
                    'url': url
                }

                content_with_citations.append(cited_answer)

    # Format content and references
    formatted_content = "\n".join([
        f"{entry['text']} {entry['citation']}" 
        for entry in content_with_citations
    ])

    formatted_urls = "\n".join([
        f"{citation_num}. {url}" 
        for citation_num, url in reference_map.items()
    ])

    return {
        "question": query,
        "content": formatted_content,
        "urls": formatted_urls,
        "reference_map": reference_map,
        "status": "successful"
    }

prompt = ChatPromptTemplate.from_messages([
    ("system", '''
You are a Biostatistics assistant named **"limmaGenie"**, an expert in performing limma (Linear Models for Microarray or Omics Data) analyses across omics types (e.g., RNA-seq, microarray, proteomics, ChIP-seq, ATAC-seq, BS-seq, Hi-C).

## Behavior Rules

- If `user_input` is exactly `"greeting"`, respond with a short greeting and **stop**.
- If the user query is **not related to limma**, return only this exact signal: `__NOT_LIMMA__` and **do not answer**.
- If **no relevant RAG context** is found:
  - Do not generate a response.
  - Return this exact signal only: `__TRIGGER_WEBSEARCH__`
- If a **web search is attempted but still no results**:
  - Return this exact signal only: `__NO_ANSWER_FOUND__`
- If returning a special keyword signal (e.g., __NOT_LIMMA__, __TRIGGER_WEBSEARCH__, __NO_ANSWER_FOUND__), 
  return the keyword **as plain text only** (not inside quotes, backticks, or code blocks).
---

## Rules for Context Use

1. **Extract relevant facts** from the provided RAG context only.
2. **Never reference or copy names or usernames** from the context. Avoid author attributions or signatures.
3. **Construct the response** using:
   - Markdown formatting.
   - Code blocks where applicable (`r` for R code).
   - Citations in `[X]` format based on context source numbers (use each source once per key point).
4. If the user requests **limma code for a multifactor design**, always combine factors with:
   ```r
   group <- paste(factor1, factor2, sep = "_")
    ```
   - Then build the design matrix using this `group`.

---

## Format Rules

- Use **Markdown** for the full response.
- Use syntax-highlighted code blocks: ```r for R code.
- Keep language clear and beginner-friendly.
- Never include a separate references section — use inline citations.

---

Now answer user's latest query using the same language the user used, 
incorporating the citations.

`CONTEXT`: {context}
'''),

    ("human", '''
`User Message`: 
{user_input} 

Reply for `User Message`. If code is requested, explain the code also in detail in a simple way, 
and include the appropriate citations from the context.
''')
])

# Create the chain
chain = prompt | llm

def get_response_llm(
    user_query: str,
    topic="limma",
    process_vector_search_results=process_vector_search_results,
    chain=chain
):
    try:
        # --- GREETING DETECTION ---
        greeting_keywords = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon",
                             "good evening", "howdy", "what's up", "hola"]
        greetings = [
            "Hi there! I am *limmaGenie*. How can I help you today?",
            "Hello! *limmaGenie* here. How may I assist you today?",
            "Hey! I'm *limmaGenie*. What can I do for you today?",
            "Greetings! I'm *limmaGenie*. How can I assist you?",
            "Hi! This is *limmaGenie*. How may I help you today?",
            "Hello! I am *limmaGenie*, here to help. What do you need assistance with?",
            "Hey there! *limmaGenie* at your service. How can I support you today?",
            "Hi! *limmaGenie* here. Let me know how I can help!",
            "Good day! I'm *limmaGenie*. How may I assist you today?",
            "Hi there! This is *limmaGenie* speaking. How can I help?"
        ]

        if user_query.lower().strip() == "greeting" or any(k in user_query.lower() for k in greeting_keywords):
            return [random.choice(greetings), "successful"]

        # --- STEP 1: VECTOR SEARCH ---
        answer = process_vector_search_results(
            query=user_query,
            embeddings_func=embeddings,
            config=CONFIG
        )

        if answer["status"] != "successful":
            return ["We are facing connection issues... \n Please try after some time.", "warning"]

        # --- STEP 2: FIRST LLM PASS ---
        input_json = {
            "user_input": user_query,
            "context": answer["content"]
        }

        llm_response = chain.invoke(input_json)
        response_content = str(llm_response.content).strip()

        # --- STEP 3: HANDLE NOT LIMMA RESPONSE  ---
        if response_content == "__NOT_LIMMA__":
            return ["Your question doesn't seem to be related to limma. Please ask me about limma analysis.", "successful"]

        # --- STEP 4: WEB SEARCH IF CONTEXT MISSING ---
        if response_content == "__TRIGGER_WEBSEARCH__":
            logger.info(f"[WEBSEARCH_TRIGGERED] Web search initiated for query: {user_query}")
            web_result = searchQuery(user_query)  # same format as vector result

            if web_result["status"] == "successful":
                input_json["context"] = web_result["content"]
                llm_response = chain.invoke(input_json)
                response_content = str(llm_response.content).strip()

                # Handle no answer after web search
                if response_content == "__NO_ANSWER_FOUND__":
                    return [f"*To Note*: We didn't find matching context. This is LLM-generated advice. "
                            f"Please consider posting your question to the Bioconductor community.", "no_context"]

                return [f"{llm_response.content}\n\nReferences:\n{web_result['urls']}", "successful"]

            else:
                return ["Web search failed. Please try again later.", "warning"]

        # --- STEP 5: NO MATCHING RESULT ANYWHERE ---
        elif response_content == "__NO_ANSWER_FOUND__":
            return [f"*To Note*: We didn't find matching context. This is LLM-generated advice. "
                    f"Please consider posting your question to the Bioconductor community.", "no_context"]

        # --- STEP 6: NORMAL SUCCESSFUL RESPONSE ---
        # urls_list = answer['urls']
        # url_temp = ""
        # for idx, url in enumerate(urls_list, 1):
        #     urls_list[idx] = url.split(",")[0] + "  " + 

        return [f"{response_content}\n\nReferences:\n{answer['urls']}", "successful"]

    except Exception as e:
        logger.error(f"Unexpected error in get_response_llm: {e}")
        return ["An unexpected error occurred.", "error"]


# Example usage (commented out)
if __name__ == "__main__":
    query = "Differential gene expression analysis on haplotype-resolved diploid assemblyedgeRDESeq2haplotypelimma" 
    answers = get_response_llm(user_query=query)
    print(answers)