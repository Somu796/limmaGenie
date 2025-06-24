import stat
import time
import streamlit as st
import random
from answers_retrieval import get_response_llm
from answers_retrieval import process_vector_search_results, chain, embeddings, CONFIG, llm
import logging
from langchain.prompts import PromptTemplate

##############################################
def generate_answer(query, context):
    prompt = PromptTemplate(
        input_variables=["query", "context"],
        template='''
Role:
You are assistant specializing in Limma (Linear Models for Microarray/Omics Data) analysis with R.
Core Instructions:
Context Usage: Use provided context as supplementary information only—not as definitive source. Answer based on the user's specific question using your expertise. Don't infer beyond what's explicitly asked.
Citations: When referencing context, use format [X] where X = document number. Use each citation once per key point. No separate references section.
Code Requirements: If code is requested, provide detailed yet simple explanations with relevant citations.
Response Framework:
Answer in user's language
Focus on the specific question asked
Integrate context citations naturally
Explain technical concepts clearly
CONTEXT: {context}
USER QUERY: {query}
TASK: Provide a focused response addressing the user's query above.
'''
    )
    # print("----------------------------------------")
    # print(context)
    # print("----------------------------------------")
    return llm.invoke(prompt.format(query=query, context=context)).content.strip()

##################################################################################
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Format web search context
def build_context(results):
    context = ""
    # for i, res in enumerate(results):
    #     if "error" in res:
    #         continue
    #     context += f"[{i+1}] Title: {res['title']}\nURL: {res['url']}\n"
    #     if res.get("question"):
    #         context += f"Question: {res['question'][0]}\n"
    #     if res.get("answers"):
    #         context += f"Answer: {res['answers'][0]}\n"
    #     context += "\n"
    # return context.strip()
    context = ''
    if results.get('content') and len(results['content']) > 0:
        for i, item in enumerate(results['content']):
            context += f"[{i+1}] {item}\n"
    return context

greetings = [
    "Hi there! I am limmaGenie. How can I help you today?",
    "Hello! limmaGenie here. How may I assist you today?",
    "Hey! I'm limmaGenie. What can I do for you today?",
    "Greetings! I'm limmaGenie. How can I assist you?",
    "Hi! This is limmaGenie. How may I help you today?",
    "Hello! I am limmaGenie, here to help. What do you need assistance with?",
    "Hey there! limmaGenie at your service. How can I support you today?",
    "Hi! limmaGenie here. Let me know how I can help!",
    "Good day! I'm limmaGenie. How may I assist you today?",
    "Hi there! This is limmaGenie speaking. How can I help?"
]

# Page config
st.set_page_config(
    page_title="limmaGenie",
    page_icon="limmaGenieLogo.jpeg",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "GPT-3.5-Turbo-0125"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize modal state
if 'show_about' not in st.session_state:
    st.session_state.show_about = False

# Clear chat function
def clear_chat():
    st.session_state.messages = []
    st.rerun()

# Inject CSS from file
with open("style_header_ui.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Replace your header layout section with this:

# Header layout - Updated version
st.markdown("""
<style>
.header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    padding: 0.5rem 0;
    margin-bottom: 1rem;
}
.header-left {
    flex: 0 0 auto;
}
.header-right {
    flex: 0 0 auto;
}
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0,0,0,0.6);
    z-index: 1000;
    backdrop-filter: blur(5px);
}
.modal-content-wrapper {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: white;
    padding: 30px;
    border-radius: 12px;
    max-width: 600px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    z-index: 1001;
}
.modal-content-wrapper h2 {
    margin-top: 0;
    color: #4B0082;
}
.modal-content-wrapper p,
.modal-content-wrapper h4 {
    text-align: justify;
}
            
.stButton {
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
}


.st-emotion-cache-1rwb540:hover {
  border: 2px solid #6B65C7; 
  color: #6B65C7;
  cursor: pointer;
  padding: 8px 16px;
  transition: all 0.1s ease;
}
            
.st-emotion-cache-1du70zn:hover {
  color: #6B65C7; 
  cursor: pointer;
  transition: all 0.1s ease;
}
.stVerticalBlock {
    padding-left: 0.5rem;
            }

.stSpinner{
    color: #6B65C7;} 

    .custom-spinner-text {
        font-size: 1.5em;
        font-weight: bold;
        background: linear-gradient(
            270deg,
            #6B65C7,
            #8E44AD,
            #9B59B6,
            #A569BD,
            #BB8FCE,
            #6B65C7
        );
        background-size: 1000% 100%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: flow 5s linear infinite;
        text-align: center;
        margin-top: 1em;
    }

    @keyframes flow {
        0% { background-position: 0% 50%; }
        100% { background-position: 100% 50%; }
    }            

</style>
""", unsafe_allow_html=True)

# Header layout with proper spacing
col_left, col_right = st.columns([1, 1])

with col_left:
    if st.button("🗑️ Clear chat"):
        clear_chat()

with col_right:
    # Create a container to align the popover to the right
    with st.container():
        # Use columns to push the popover to the right
        spacer, popover_col = st.columns([3, 1])
        with popover_col:
            with st.popover("ℹ️ About us", use_container_width=False):
                # Modal content
                st.markdown("""
### 🌟 Welcome to limmaGenie!

**limmaGenie** is your friendly, web-based assistant designed to help you perform differential expression analysis with the powerful _limma_ package. Whether you're just getting started or already an experienced researcher looking to validate your approach, _limmaGenie_ is here to guide you.

From **RNA-seq** and **microarrays** to **proteomics** and other high-throughput omics data, _limmaGenie_ walks you through every step of the workflow—helping you design your experiment, generate accurate R code, and interpret your results with confidence.

### 📘 Want to Learn More About _limma_?

Here are some great starting points if you'd like to dive deeper into how `limma` works:

- [limma User's Guide (Bioconductor)](https://bioconductor.org/packages/release/bioc/vignettes/limma/inst/doc/usersguide.pdf) 
- [A guide to creating design matrices for gene expression experiments](https://f1000research.com/articles/9-1444/v1)
- [RNA-seq analysis is easy as 1-2-3 with limma, Glimma and edgeR](https://f1000research.com/articles/5-1408/v2)

---

### 👨‍🔬 Built & Maintained By

This project is actively developed and maintained by:

- **[Sudipta Kumar Hazra](https://www.linkedin.com/in/sudipta-kumar-hazra/)** – MSc (Research) Student at **UCC and Teagasc, Ireland**
- **[Kushagra Bhatnagar](https://www.linkedin.com/in/kushagra-bhatnagar-aa8120219/)** – Machine Learning Developer | Specializing in LLMs & NLP

We're committed to making omics analysis **simpler**, **smarter**, and more **accessible** to everyone in the research community.

---

### 💬 Got Suggestions?

We’d love to hear from you! Your feedback helps us grow and improve.  
[Submit feedback here](https://docs.google.com/forms/d/e/1FAIpQLSdfkM5M5M9NwqMNBUT0_Z41gS6zlSDYpXvFb02gWi9GoplFbg/viewform)
""")





# Title and subtitle with proper spacing
st.markdown("""
<div class="main-content">
    <h1 class="main-title">limmaGenie</h1>
    <p class="subtitle">Ask me anything about <a href="#">limma analysis</a>!</p>
</div>
""", unsafe_allow_html=True)

# Chat UI
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display chat messages with inline styles for proper alignment
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        # User message - aligned to the right
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end; width: 100%; margin-bottom: 15px;">
            <div style="background-color: #6B65C7; color: white; padding: 12px 16px; border-radius: 18px; max-width: 70%; word-wrap: break-word;">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Assistant message - aligned to the left
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start; width: 100%; margin-bottom: 15px;">
            <div style="background-color: #f1f3f4; color: #333; padding: 12px 16px; border-radius: 18px; max-width: 70%; word-wrap: break-word; border: 1px solid #e0e0e0;">
                {msg["content"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history and immediately rerun to show it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Check if we need to generate a response (last message is from user)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    mode = "vector_search"  # Set mode to vector search
    if mode == "vector_search":
        try:
            user_query = st.session_state.messages[-1]["content"]
            # Perform vector search
            with st.spinner("limmaGenie is thinking..."):
                result_list = get_response_llm(user_query)
                result_dict = {"status" : result_list[1],
                               "response" : result_list[0]}
                
                status = result_dict.get("status", "error_no_status")
                response = result_dict.get("response", "I apologize, but I encountered an error while processing your request. Please try again.")
                if status not in ["successful", "warning", "no_context"]:
                    response = "I apologize, but I encountered an error while processing your request. Please try again."
                



                
        #         answer = process_vector_search_results(
        #             query=st.session_state.messages[-1]["content"], 
        #             embeddings_func=embeddings,
        #             config=CONFIG
        #         )
            
        #     with st.spinner("Evaluating context..."):
        #         # Process successful search
        #         # print(answer["status"])
        #         # print(answer["content"])
        #         # print(answer)

        #         if answer.get("status") and answer["status"] in ["successful", "warning", "no_context"]:
        #             input_json = {
        #                 "user_input": st.session_state.messages[-1]["content"],
        #                 "context": answer["content"]
        #             }

        #             ## Check if content relevant?
        #             is_context_relevant = chain_check.invoke(input_json)
        #         else:
        #             response = "We are facing connection issues... \n Please try after some time."
        # except Exception as e:
        #     with st.spinner("Error during vector search: {e}"):
        #         time.sleep(2)
        #     logger.error(f"Error during vector search: {e}")
        #     response = "I apologize, but I encountered an error while processing your request. Please try again."


        # if is_context_relevant and is_context_relevant.content.lower() == "true":
            
        #     with st.spinner("limmaGenie generating response from existing database..."):
        #         # Invoke LLM chain
        #         llm_response = chain.invoke(input_json)
                
        #         # Combine response with references
        #         response = f"{llm_response.content}\n\nReferences from database:\n{answer['urls']}"

        # else:

        #     with st.spinner("Searching web for relevant context..."):

        #         user_query = st.session_state.messages[-1]["content"]
        #         # Show spinner and generate response
        #         try:
        #             results = combined_search(user_query)
        #             print(results)
        #             if results and results.get('status') in ["successful", "warning", "no_context"]:
        #                 context = build_context(results)
        #                 if len(context)>0:
        #                     final_answer = chain.invoke(input_json).content
        #                     # final_answer = generate_answer(user_query, context)
        #                 else:
        #                     final_answer = llm.invoke(user_query).content.strip()
                        
        #                 # Build response with sources
        #                 response = final_answer
        #                 urls = results.get('urls', [])
        #                 if len(urls)>0:
        #                     response_source = f"\n\n**Sources (from web search):**\n"
        #                     for i, url in enumerate(urls):
        #                         response_source += f"\n[{i+1}]: {url}"
        #                     # sources = [f"[{res['title']}]({res['url']})" for res in results if "error" not in res]
        #                 if response_source:
        #                     response += response_source
        #             else:
        #                 response = f"I apologize, but I encountered an error while processing your request during web search. Please try again."
        #                 # response = "I couldn't find specific information for your query. Let me try to help based on my knowledge of LIMMA and Bioconductor."
        #                 # response += "\n\n" + llm.invoke(user_query).content.strip()
                        
        except Exception as e:
            response = f"I apologize, but I encountered an error while processing your request during web search. Please try again.\n\nError: {str(e)}"

            
        stream =  [response]
            
        # except Exception as e:
        #     logger.error(f"Unexpected error in get_response_llm: {e}")
        #     stream = ["An unexpected error occurred.", "error"]
        
        response_text = ""
        
        # Collect the full response
        for chunk in stream:
            response_text += chunk
    
    # elif mode == "web_search":
    #     user_query = st.session_state.messages[-1]["content"]
    #     # Show spinner and generate response
    #     with st.spinner("limmaGenie is thinking..."):
    #         try:
    #             results = combined_search(user_query)
    #             if results and isinstance(results, list):
    #                 context = build_context(results)
    #                 if context:
    #                     final_answer = generate_answer(user_query, context)
    #                 else:
    #                     final_answer = llm.invoke(user_query).content.strip()
                    
    #                 # Build response with sources
    #                 response_text = final_answer
    #                 sources = [f"[{res['title']}]({res['url']})" for res in results if "error" not in res]
    #                 if sources:
    #                     response_text += f"\n\n**Sources:**\n" + "\n".join(sources)
    #             else:
    #                 response_text = "I couldn't find specific information for your query. Let me try to help based on my knowledge of LIMMA and Bioconductor."
    #                 response_text += "\n\n" + llm.invoke(user_query).content.strip()
                    
    #         except Exception as e:
    #             response_text = f"I apologize, but I encountered an error while processing your request. Please try again.\n\nError: {str(e)}"
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    # Rerun to show the response
    st.rerun()