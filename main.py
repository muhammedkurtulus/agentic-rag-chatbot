import streamlit as st
import os
import uuid
import pdfplumber
import random
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SearchParams, PointStruct
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from crewai import Crew, Agent, Task, LLM, Process
from crewai.tools import tool
from crewai.tools import BaseTool
from typing import Type, Any
from pydantic import BaseModel, Field
import ollama
import json

load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

OLLAMA_BASE_URL = "http://localhost:11434"

embedding_model = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_BASE_URL)

crewai_llm = LLM(model="ollama/llama3.1", base_url=OLLAMA_BASE_URL)

qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)


# ----------------------------------------
# 🔧 FUNCTIONS
# ----------------------------------------
def load_pdf(pdf_file):
    """Reads the PDF file page by page and returns only the plain text."""
    full_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text


def chunk_text(text, chunk_size=256, overlap=64):
    """Splits the entire text into chunks. No metadata like page number is added."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    return [{"text": chunk} for chunk in chunks]


def get_embedding(text):
    """Embeds the text using Ollama's bge-m3 model."""

    if not text or not isinstance(text, str) or text.strip() == "":
        print("⚠️ ERROR: An invalid text was provided to the get_embedding function.")
        return None

    try:
        response = ollama.embeddings(
            model="bge-m3",
            prompt=text,
        )
        return response["embedding"]
    except Exception as e:
        print(f"🚨 Ollama Embedding Error: {e}")
        return None


def save_to_qdrant(chunks):
    """Converts the list of text chunks into embeddings and stores them in Qdrant."""
    collection = st.session_state.get("collection_name", None)
    if not collection:
        st.warning("⚠️ Please create and select a Qdrant collection first.")
        return

    points = []
    for chunk in chunks:
        text = chunk["text"]
        embedding = get_embedding(text)
        if embedding:
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), vector=embedding, payload={"text": text}
                )
            )

    if points:
        qdrant_client.upsert(collection_name=collection, points=points)
        print(
            f"✅ {len(points)} chunks have been uploaded to Qdrant. Collection: {collection}"
        )


# ----------------------------------------
# 🔍 QUERY PROCESSING FUNCTIONS
# ----------------------------------------
def process_query(user_input):
    """
    Main query processing function. Takes a user query and processes it through all steps to return a response.
    """
    result = {
        "routing": None,
        "rewritten": None,
        "retrieved": None,
        "evaluation": None,
        "answer": None,
        "error": None,
        "language": "en",  # Default language
    }

    try:
        # 1. Initial Query Analysis (Language and Greeting Detection)
        try:
            analyzer_crew = Crew(
                agents=[initial_query_analyzer_agent],
                tasks=[initial_query_analysis_task],
                verbose=True,
                process=Process.sequential,
            )
            analysis_output = analyzer_crew.kickoff(inputs={"query": user_input})
            analysis_json_str = str(analysis_output)  # Ensure it's a string
            print(f"🕵️ Initial Analysis Raw Output: {analysis_json_str}")

            # Attempt to strip any markdown or extra formatting if present
            if analysis_json_str.startswith("```json"):
                analysis_json_str = analysis_json_str.strip("```json\n")
                analysis_json_str = analysis_json_str.strip("\n```")
            analysis_json_str = analysis_json_str.strip()  # General stripping

            print(f"🕵️ Cleaned Initial Analysis JSON String: {analysis_json_str}")
            analysis = json.loads(analysis_json_str)
            query_type = analysis.get("type", "query")
            detected_language = analysis.get("language", "en")
            result["language"] = detected_language.strip().lower()
            print(f"🔤 Detected Language: {result['language']}")

            if query_type == "greeting":
                result["routing"] = "greeting"
                result["rewritten"] = user_input  # No rewrite for greetings
                result["retrieved"] = "greeting_response"
                result["evaluation"] = (
                    "yes"  # Greetings are always useful for a response
                )
                print(
                    f"👋 Greeting detected. Language: {result['language']}. Skipping further processing."
                )
                # The stream_answer_from_ollama function will handle generating the greeting message.
                # No need to return early, let the main flow continue to stream_answer_from_ollama.
            else:
                # This is a non-greeting query, proceed to tool routing
                result["routing"] = None  # To be set by tool_router_agent

        except Exception as e:
            print(
                f"Error during initial query analysis: {str(e)}. Proceeding with defaults."
            )
            result["language"] = "en"
            # Assume it's not a greeting and proceed to routing if analysis fails
            query_type = "query"

        # Proceed only if not a greeting that has been fully handled
        if result["routing"] != "greeting":
            # 2. Tool Routing step (only if not a greeting)
            try:
                # Call the undecorated Python function directly
                routing_decision = _get_similarity_routing_decision(user_input)
                result["routing"] = (
                    routing_decision.strip().lower()
                    if routing_decision
                    else "combined_search"
                )
                print(f"🔄 Direct Routing Decision: {result['routing']}")
            except Exception as e:
                print(f"Error during direct routing logic call: {str(e)}")
                result["routing"] = (
                    "combined_search"  # Default to combined search on error
                )

            # 3. Rewrite step
            try:
                result["rewritten"] = rewrite_query(user_input, result["routing"])
                print(f"✏️ Rewritten Query: {result['rewritten']}")
            except Exception as e:
                print(f"Error during rewriting: {str(e)}")
                result["rewritten"] = user_input  # Use original query on error

            # 4. Retrieve step
            try:
                result["retrieved"] = retrieve_information(
                    result["rewritten"], result["routing"]
                )
                print(f"📦 Retrieved Result: {result['retrieved'][:200]}...")
            except Exception as e:
                print(f"Error during retrieval: {str(e)}")
                result["retrieved"] = (
                    "Unable to retrieve information. Please try again with a different query."
                )

            # 5. Evaluate step
            try:
                result["evaluation"] = evaluate_results(
                    result["rewritten"], result["retrieved"]
                )
                print(f"🧠 Evaluator Decision: {result['evaluation']}")
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                result["evaluation"] = "no"  # Default on error

        return result
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        result["error"] = str(e)
        return result


def rewrite_query(query, routing):
    """
    Rewrites the query or returns it unchanged based on routing decision.
    """
    if routing == "vector_store":
        # We don't modify the query for vector search
        return query
    else:
        # We expand the query for web search
        try:
            rewrite_crew = Crew(
                agents=[rewrite_agent],
                tasks=[rewrite_task],
                verbose=True,
                process=Process.sequential,
            )
            rewritten_result = rewrite_crew.kickoff(
                inputs={"query": query, "routing": routing}
            )
            return str(rewritten_result)
        except Exception as e:
            print(f"Error in rewrite_query: {e}")
            return query  # Return original query on error


def get_vector_search_results(query, limit=3):
    """
    Retrieves the most relevant results from vector database for the given query.

    Args:
        query (str): Search query
        limit (int): Number of results to retrieve

    Returns:
        str: Processed text results, returns empty string if no results found
    """
    collection = st.session_state.get("collection_name", None)
    if not collection:
        print("⚠️ No collection selected.")
        return ""

    query_vector = embedding_model.embed_query(query)
    search_result = qdrant_client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=limit,
        search_params=SearchParams(hnsw_ef=64, exact=False),
    )

    if not search_result:
        print("No results found from vector search.")
        return ""

    # Convert results to readable format
    relevant_chunks = []
    for i, hit in enumerate(search_result):
        text = hit.payload["text"]
        text = text.replace("\n\n", " ").replace("\n", " ")
        relevant_chunks.append(f"Chunk {i+1}: {text}")
        print(f"📄 Retrieved Chunk {i+1}: {text[:100]}...")

    retrieved = "\n\n".join(relevant_chunks)
    return retrieved


def retrieve_information(query, routing):
    """
    Retrieves information using the appropriate strategy based on routing decision.
    """
    try:
        # Handle greeting case directly
        if routing == "greeting":
            return "greeting_response"

        elif routing == "vector_store":
            # First try direct vector search
            try:
                retrieved = get_vector_search_results(query)

                if not retrieved or retrieved.strip() == "":
                    raise ValueError("Empty results from vector search.")

                return retrieved

            except Exception as e:
                print(f"Error in direct vector search: {e}")
                # Fall back to agent if direct search fails
                retriever_crew = Crew(
                    agents=[retriever_agent],
                    tasks=[retriever_task],
                    verbose=True,
                    process=Process.sequential,
                )
                retrieved_result = retriever_crew.kickoff(
                    inputs={"query": query, "routing": routing, "search_type": routing}
                )
                return str(retrieved_result)

        elif routing == "combined_search":
            # For combined search, we'll do both vector and web search
            try:
                # First try vector search
                vector_results = get_vector_search_results(query)

                # Then do web search
                search = TavilySearchResults()
                web_results = search.run(query)
                print(f"🌐 Web Search Results: {str(web_results)}")

                # Combine results
                combined_results = ""
                if vector_results:
                    combined_results += (
                        "Vector Search Results:\n" + vector_results + "\n\n"
                    )
                combined_results += "Web Search Results:\n" + str(web_results)

                return combined_results

            except Exception as e:
                print(f"Error in combined search: {e}")
                # Fall back to web search only
                try:
                    search = TavilySearchResults()
                    web_results = search.run(query)
                    print(f"🌐 Web Search Results (Fallback): {str(web_results)}")
                    return "Web Search Results:\n" + str(web_results)
                except Exception as web_error:
                    print(f"Error in web search fallback: {web_error}")
                    return "Unable to retrieve information. Please try again with a different query."

        else:  # web_search or any other value (for backward compatibility)
            # Use retriever agent for web search
            retriever_crew = Crew(
                agents=[retriever_agent],
                tasks=[retriever_task],
                verbose=True,
                process=Process.sequential,
            )
            retrieved_result = retriever_crew.kickoff(
                inputs={
                    "query": query,
                    "routing": "web_search",
                    "search_type": "web_search",
                }
            )
            return str(retrieved_result)

    except Exception as e:
        print(f"Error in retrieve_information: {e}")
        # Use direct tools as last resort
        try:
            search = TavilySearchResults()
            web_results = search.run(query)
            print(f"🌐 Web Search Results (Last Resort): {str(web_results)}")
            return str(web_results)
        except Exception as fallback_error:
            print(f"Error in retrieve_information fallback: {fallback_error}")
            return "Unable to retrieve information. Please try again with a different query."


def get_similarity_score(query):
    """
    Calculates the highest similarity score for the specified query.

    Args:
        query (str): Query to calculate similarity score for

    Returns:
        float: Calculated similarity score, returns 0.0 in case of error
    """
    print(f"Running similarity calculation with query: '{query}'")

    collection = st.session_state.get("collection_name", None)
    if not collection:
        print("⚠️ No collection selected.")
        return 0.0

    try:
        query_vector = embedding_model.embed_query(query)

        search_results = qdrant_client.search(
            collection_name=collection, query_vector=query_vector, limit=3
        )

        if not search_results:
            print("No results found.")
            return 0.0

        best_match = max(search_results, key=lambda x: x.score)
        score = float(best_match.score)
        print(f"Highest similarity score: {score}")
        return score
    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return 0.0


def get_similarity_based_routing(query):
    """
    Determines the routing destination based on similarity score calculation.

    Args:
        query (str): Query to be routed

    Returns:
        str: Returns either "vector_store" or "combined_search"
    """
    similarity_score = get_similarity_score(query)

    if similarity_score > 0.40:
        return "vector_store"
    else:
        return "combined_search"


def evaluate_results(query, retrieved):
    """
    Evaluates the usefulness of retrieved results.
    """
    try:
        evaluator_crew = Crew(
            agents=[evaluator_agent],
            tasks=[evaluator_task],
            verbose=True,
            process=Process.sequential,
        )
        evaluation_result = evaluator_crew.kickoff(
            inputs={"query": query, "retrieved": retrieved}
        )
        evaluation = str(evaluation_result).strip().lower()

        # Handle invalid evaluations
        if evaluation not in ["yes", "no"]:
            print(f"Invalid evaluation: '{evaluation}', defaulting to 'no'")
            evaluation = "no"

        return evaluation
    except Exception as e:
        print(f"Error in evaluate_results: {e}")
        return "no"  # Return default value on error


# ----------------------------------------
# 🧰 TOOLS
# ----------------------------------------
class TavilySearchInput(BaseModel):
    """The query required to perform a web search using the Tavily API."""

    query: str = Field(..., description="The search query for Tavily Web Search")


class TavilySearchTool(BaseTool):
    name: str = "Tavily Web Search"
    description: str = (
        "Uses Tavily API to perform a web search and return relevant results."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    def _run(self, query: str) -> Any:
        search = TavilySearchResults()
        web_results = search.run(query)
        print(f"🌐 Web Search Tool Results (TavilySearchTool): {str(web_results)}")
        return web_results


@tool("Qdrant Similarity Retriever")
def retrieve_highest_similarity(query: str) -> float:
    """
    Retrieves the highest similarity score from Qdrant based on the input query.
    Converts the query to a vector using Ollama embeddings and searches the Qdrant collection.
    Returns the highest similarity score among the top results as a float value.
    """
    return get_similarity_score(query)


@tool("Qdrant Vector Search")
def search_qdrant_tool(query: str) -> str:
    """Fetches the most relevant text chunks from Qdrant vector database based on semantic similarity with the input query."""
    result = get_vector_search_results(query)
    if not result:
        return "No relevant information found."
    return result


web_search_tool = TavilySearchTool()
# retrieve_highest_similarity_tool = retrieve_highest_similarity
vector_search_tool = search_qdrant_tool


# Undecorated Python function for routing logic
def _get_similarity_routing_decision(query: str) -> str:
    """Calculates similarity for the query using get_similarity_score.
    If score > 0.40, returns 'vector_store'.
    Else (score <= 0.40 or error), returns 'combined_search'."""
    score_val = 0.0  # Default score for logging in case of early error
    decision = "combined_search"  # Default decision
    try:
        score_val = get_similarity_score(query)  # existing function
        if score_val > 0.40:
            decision = "vector_store"
        # else decision remains "combined_search"
    except Exception as e:
        print(
            f"Error in _get_similarity_routing_decision: {e}. Defaulting to combined_search."
        )
        # decision is already "combined_search"
    print(f"Similarity Based Routing Logic: score={score_val}, decision='{decision}'")
    return decision


# Decorated tool (currently not used by an agent but defined for potential future use)
# @tool("Similarity Based Router Tool")
# def similarity_based_router_tool_for_agent(query: str) -> str:
#     """CrewAI Tool wrapper for similarity based routing decision.
#     Calculates similarity for the query and returns 'vector_store' or 'combined_search'.
#     """
#     return _get_similarity_routing_decision(query)


# ----------------------------------------
# 🤖 AGENTS & TASKS
# ----------------------------------------

# Initial Query Analyzer Agent
initial_query_analyzer_agent = Agent(
    role="Initial Query Analyzer",
    goal="Analyze the user's query to detect its language and determine if it is a greeting. Output the analysis as a structured JSON.",
    backstory=(
        """Expert in multilingual linguistic analysis, capable of identifying greetings in various languages
    and discerning the primary language of any text. Provides clear, structured output for downstream processing."""
    ),
    llm=crewai_llm,
    allow_delegation=False,
    verbose=True,
    max_retry_limit=2,
    max_iterations=3,
    max_execution_time=30,
)


# Rewrite Agent
rewrite_agent = Agent(
    role="Query Expansion Specialist",
    goal="Enhance and expand the search query to retrieve more relevant results",
    backstory=(
        """An expert in search query optimization, NLP, and semantic expansion.
    Uses advanced techniques like synonym enrichment, contextual expansion, and 
    domain-specific augmentation to improve search effectiveness."""
    ),
    llm=crewai_llm,
    allow_delegation=False,
    verbose=True,
    max_retry_limit=2,
    max_iterations=3,
    max_execution_time=60,
)


# Retriever Agent
retriever_agent = Agent(
    role="Information Retriever",
    goal="Fetch relevant information from appropriate source",
    backstory="""Experienced in efficient information retrieval from multiple
    sources with expertise in both web and vector store searches.""",
    llm=crewai_llm,
    allow_delegation=False,
    verbose=True,
    max_retry_limit=3,
    max_iterations=4,
    max_execution_time=90,
)


# Evaluator Agent
evaluator_agent = Agent(
    role="Content Evaluator",
    goal="Assess the relevance and completeness of retrieved information",
    backstory="""Expert in content analysis and quality assessment with
    strong analytical skills and attention to detail.""",
    llm=crewai_llm,
    allow_delegation=False,
    verbose=True,
    max_retry_limit=2,
    max_iterations=3,
    max_execution_time=60,
)


# Task for Initial Query Analyzer Agent
initial_query_analysis_task = Task(
    description=(
        """Analyze the user's EXACT query: "{query}"
    1. Determine the primary language of the query. Return it as a two-letter ISO 639-1 code (e.g., 'en', 'tr', 'es'). If unsure, default to 'en'.
    2. Determine if the query is a greeting (e.g., "hello", "merhaba", "hola", "hi").
    
    Respond with ONLY a JSON string in ONE of the following formats:
    - If it IS a greeting: {{"type": "greeting", "language": "language_code"}} (e.g., {{"type": "greeting", "language": "tr"}})
    - If it IS NOT a greeting: {{"type": "query", "language": "language_code"}} (e.g., {{"type": "query", "language": "en"}})
    
    DO NOT include any other text or explanation."""
    ),
    expected_output=(
        'A JSON string. Example for greeting: {{"type": "greeting", "language": "en"}}. Example for non-greeting: {{"type": "query", "language": "es"}}'
    ),
    agent=initial_query_analyzer_agent,
    # No tools needed for this agent, relies on LLM's inherent understanding
)


# Task for Rewrite Agent
rewrite_task = Task(
    description="""Strictly follow these instructions for analyzing the query: {query}, based on routing decision: {routing}.

    IF routing decision is 'vector_store':
    - Return ONLY the original query UNCHANGED, with NO additional analysis or expansion.
    - DO NOT question or analyze the routing decision.
    
    IF AND ONLY IF routing decision is 'web_search' OR routing decision is 'combined_search':
    - Expand the query using techniques like synonym expansion, contextual expansion, etc.
    - Make sure the expansion is relevant and coherent.
    
    NO OTHER TEXT OR ANALYSIS should be included in your response.
    """,
    expected_output="""The original query string if routing is 'vector_store', or an expanded query if routing is 'web_search' or 'combined_search'.""",
    agent=rewrite_agent,
)


# Task for Retriever Agent
retriever_task = Task(
    description="""Use ONLY the search tool that matches the router's decision:
    - If routing is 'vector_store', use ONLY vector_search_tool
    - If routing is 'web_search', use ONLY web_search_tool
    
    IMPORTANT: Do not override this decision or use both tools.
    
    After retrieving information:
    1. Organize the information into a clear, concise summary
    2. Focus on the most relevant parts that directly answer the query
    3. For specific questions, EXTRACT AND HIGHLIGHT exact details related to what was asked
    4. For "who is" queries, focus mainly on profession/title
    5. Ensure all relevant information related to the specific query is preserved
    6. If specific information is not found, clearly note this
    """,
    expected_output="""A well-formatted, concise summary of the retrieved information that directly 
    addresses the query. The response should be clear, coherent, and focused on the most relevant details.""",
    agent=retriever_agent,
    tools=[vector_search_tool, web_search_tool],
    context=[rewrite_task],
)


# Task for Evaluator Agent
evaluator_task = Task(
    description="""Analyze the retrieved information to determine if it's relevant and useful for answering the query.
    
    IMPORTANT: 
    1. Your task is ONLY to determine if the retrieved information is USEFUL for answering the query.
    2. RESPOND WITH EXACTLY ONE WORD: EITHER 'yes' OR 'no'.
    3. Say 'yes' if there is ANY useful information, even if partial.
    4. Say 'no' ONLY if completely irrelevant.
    5. Basic biographical information (profession, title) for 'who is' questions IS useful.
    
    YOUR OUTPUT MUST BE ONLY 'yes' OR 'no'. NO OTHER TEXT.""",
    expected_output="""ONLY the single word 'yes' or 'no'.""",
    agent=evaluator_agent,
    context=[retriever_task],
)


# ----------------------------------------
# 🧠 LLM
# ----------------------------------------
def stream_answer_from_ollama(rewritten, retrieved, evaluation, language="en"):
    # Handle greeting case
    if retrieved == "greeting_response":
        # Use different greetings based on detected language
        greetings = {
            "en": [
                "Hello! How can I help you?",
                "Hi there! How can I assist you today?",
                "Greetings! Feel free to ask me any question.",
                "Hello! You can ask me anything.",
            ],
            "tr": [
                "Merhaba! Size nasıl yardımcı olabilirim?",
                "Selam! Bugün size nasıl yardımcı olabilirim?",
                "Merhaba! Bana istediğiniz soruyu sorabilirsiniz.",
                "Selam! Bana herhangi bir konuda soru sorabilirsiniz.",
            ],
            # Add more languages as needed
            "default": [
                "Hello! How can I help you?",
                "Hi there! How can I assist you today?",
            ],
        }

        # Use language from the router agent
        print(f"🔤 Using Language for Greeting: {language}")
        if language in greetings:
            greeting_response = random.choice(greetings[language])
        else:
            greeting_response = random.choice(greetings["default"])

        # Return the greeting without streaming
        yield greeting_response
        return greeting_response

    # Always treat partial information as useful
    if evaluation not in ["yes", "no"]:
        print(f"Invalid evaluation: '{evaluation}', defaulting to 'yes'")
        evaluation = "yes"  # Default to yes for invalid responses

    # Get language from the router agent
    print(f"🔤 Using Language for Response: {language}")
    detected_lang = language

    prompt = f"""Query: {rewritten}

Retrieved Information:
{retrieved}

Evaluation: {evaluation}

This is a biographical query system. Follow these guidelines:
1. For "who is" questions: ONLY give a very brief answer (1-2 sentences) about profession 
2. For SPECIFIC questions: FOCUS ONLY on that specific information requested in the query
3. If the specific information is not found, clearly state that this information is not available
"""

    system_prompt = f"""This system is designed to provide information about people, with two distinct response types:

    INSTRUCTIONS:
    1. Only use information from the retrieved data
    2. For "who is" questions: provide ONLY a SHORT answer with profession/title (1-2 sentences)
    3. For SPECIFIC questions: FOCUS EXCLUSIVELY on answering that specific question with ONLY relevant details
    4. Each specific question type should get a targeted, relevant answer about ONLY what was asked
    5. If specific information is not in the data, say that this information is not available
    6. NEVER extrapolate or guess missing information
    7. RESPOND IN THE SAME LANGUAGE AS THE QUERY: the detected language is {detected_lang}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Stream the response from ollama
    try:
        response = ollama.chat(
            model="llama3.1",
            messages=messages,
            stream=True,
        )

        full_text = ""
        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                full_text += content
                yield content

        return full_text
    except Exception as e:
        error_message = f"🚨 Error generating response: {e}"
        print(error_message)
        yield error_message
        return error_message
