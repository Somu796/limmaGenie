from langchain_community.tools import DuckDuckGoSearchResults
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
import time
import random

# Set up DuckDuckGo search tool
search = DuckDuckGoSearchResults(output_format="list", max_results=5)

# --- Helpers ---

def function_to_filter_bioconductor_links(websearch_output):
    try:
        return [
            urlparse(entry['link']).path
            for entry in websearch_output
            if "https://support.bioconductor.org/" in entry['link']
        ]
    except Exception:
        return []

def clean_text(text):
    if isinstance(text, str):
        return text.replace("\n", " ").replace("\\'", "'").replace("\\", "").replace("\xa0", "")
    return text

def clean_after_session_info(text):
    if not isinstance(text, str):
        return text
    idx = text.lower().find('sessioninfo()')
    return text[:idx].strip() if idx != -1 else text

# --- Scraping Bioconductor Answers ---

def scrape_question_answers(question_links):
    all_data = []

    for link in question_links:
        url = f"https://support.bioconductor.org{link}"
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract question and discussion
            question = []
            discussion = []

            segments = soup.find_all("div", class_="ui vertical segment")
            for i, segment in enumerate(segments):
                text_block = segment.find("span", itemprop="text")
                if text_block:
                    cleaned = clean_after_session_info(clean_text(text_block.get_text()))
                    if i == 0:
                        question.append(cleaned)
                    else:
                        discussion.append(cleaned)

            # Store result
            page_data = {
                "url": url,
                "title": soup.find("div", class_="title").get_text(strip=True) if soup.find("div", class_="title") else "No title",
                "question": question,
                "answers": discussion,
            }

            all_data.append(page_data)
            time.sleep(round(0.1 + random.uniform(0, 0.3), 2))

        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    return all_data

# --- Combined Search + Format for RAG ---

def searchQuery(query):
    try:
        websearch_output = search.invoke(query)
        filtered_links = function_to_filter_bioconductor_links(websearch_output)

        if not filtered_links:
            return {
                "content": [],
                "urls": [],
                "status": "failed",
                "error": "No relevant Bioconductor results found."
            }

        data = scrape_question_answers(filtered_links)

        content_blocks = []
        source_urls = []

        for item in data:
            if item.get("question") or item.get("answers"):
                block = ""
                if item["question"]:
                    block += f"Q: {item['question'][0]}\n"
                if item["answers"]:
                    block += "\n".join(f"A: {a}" for a in item["answers"])
                content_blocks.append(block)
                source_urls.append(item["url"])

        return {
            "content": content_blocks,
            "urls": source_urls,
            "status": "successful"
        }

    except Exception as e:
        return {
            "content": [],
            "urls": [],
            "status": "failed",
            "error": str(e)
        }
