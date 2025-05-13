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

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

VECTOR_SEARCH_LIMIT = 3  # Number of results to retrieve from vector search

embedding_model = OllamaEmbeddings(model="bge-m3", base_url=OLLAMA_BASE_URL)

crewai_llm = LLM(model="ollama/llama3.1", base_url=OLLAMA_BASE_URL)

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


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
        "language": "en",
    }

    try:
        # 1. Initial Query Analysis (Language and Social Expression Detection)
        try:
            analyzer_crew = Crew(
                agents=[initial_query_analyzer_agent],
                tasks=[initial_query_analysis_task],
                verbose=True,
                process=Process.sequential,
            )
            analysis_output = analyzer_crew.kickoff(inputs={"query": user_input})
            analysis_json_str = str(analysis_output)
            print(f"🕵️ Initial Analysis Raw Output: {analysis_json_str}")

            # Attempt to strip any markdown or extra formatting if present
            if analysis_json_str.startswith("```json"):
                analysis_json_str = analysis_json_str.strip("```json\n")
                analysis_json_str = analysis_json_str.strip("\n```")
            analysis_json_str = analysis_json_str.strip()

            print(f"🕵️ Cleaned Initial Analysis JSON String: {analysis_json_str}")
            initial_analysis = json.loads(analysis_json_str)
            detected_language = initial_analysis.get("language", "en").strip().lower()
            translated_query = initial_analysis.get("translated_query", user_input) # Use original if translation fails
            result["language"] = detected_language
            print(f"🔤 Detected Language: {result['language']}")
            print(f"🌐 Translated Query: {translated_query}")

            # Pass the translated query and detected language to the simplifier/classifier
            user_input_for_simplifier = translated_query # Use translated query for next steps

        except Exception as e:
            print(
                f"Error during initial query analysis: {str(e)}. Proceeding with defaults."
            )
            result["language"] = "en"
            detected_language = "en" # default
            translated_query = user_input # default
            user_input_for_simplifier = user_input # Use original query if analysis fails
            # Assume it's not a social expression and proceed to routing if analysis fails
            query_type = "query"


        # 2. Query Simplification and Social Classification step
        simplified_query_data_str = ""
        try:
            simplifier_crew = Crew(
                agents=[query_simplifier_agent],
                tasks=[query_simplifier_task],
                verbose=True,
                process=Process.sequential,
            )
            simplified_output_raw = simplifier_crew.kickoff(
                inputs={"translated_query": user_input_for_simplifier, "detected_language": detected_language}
            )
            simplified_query_data_str = str(simplified_output_raw).strip()
            print(f"🔍 Simplified/Classified Query Raw Output: {simplified_query_data_str}")

            if simplified_query_data_str.startswith("```json"):
                simplified_query_data_str = simplified_query_data_str.strip("```json\n")
                simplified_query_data_str = simplified_query_data_str.strip("\n```")
            simplified_query_data_str = simplified_query_data_str.strip()
            print(f"🔍 Cleaned Simplified/Classified Query JSON String: {simplified_query_data_str}")

            simplified_query_data = json.loads(simplified_query_data_str)
            query_type = simplified_query_data.get("type", "query")
            processed_query_output = simplified_query_data.get("output", user_input_for_simplifier)

            if query_type == "social":
                result["routing"] = "social"
                result["answer"] = processed_query_output # This is the social response in original language
                result["rewritten"] = user_input # Original user input
                result["retrieved"] = "social_response"
                result["evaluation"] = "yes"
                print(
                    f"👋 Social expression detected by simplifier. Language: {result['language']}. Response: {result['answer']}. Skipping further processing."
                )
                return result
            else:
                # This is a non-social expression query, use the simplified output for further processing
                user_input = processed_query_output # The simplified query
                result["routing"] = None # To be set by tool_router_agent
                print(f"💡 Query classified as '{query_type}'. Simplified Query: {user_input}")

        except Exception as e:
            print(
                f"Error during query simplification/classification: {str(e)}. Using translated/original query for routing."
            )
            user_input = user_input_for_simplifier # Fallback to translated (or original if translation failed)
            # Assume it's a query and proceed to routing if simplification fails
            query_type = "query"


        # Proceed only if not a social expression that has been fully handled
        if result["routing"] != "social":
            # 3. Tool Routing step (only if not a social expression)
            try:
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

            # 4. Rewrite step
            try:
                result["rewritten"] = rewrite_query(user_input, result["routing"])
                print(f"✏️ Rewritten Query: {result['rewritten']}")
            except Exception as e:
                print(f"Error during rewriting: {str(e)}")
                result["rewritten"] = user_input  # Use original query on error

            # 5. Retrieve step
            try:
                result["retrieved"] = retrieve_information(
                    result["rewritten"], result["routing"]
                )
            except Exception as e:
                print(f"Error during retrieval: {str(e)}")
                result["retrieved"] = (
                    "Unable to retrieve information. Please try again with a different query."
                )

            # 6. Evaluate step
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


def get_vector_search_results(query):
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
        limit=VECTOR_SEARCH_LIMIT,
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
        # Handle social expression case directly
        if routing == "social":
            return "social_response"

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
            collection_name=collection,
            query_vector=query_vector,
            limit=VECTOR_SEARCH_LIMIT,
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

# Decorated tool (currently not used by an agent but defined for potential future use)
# @tool("Similarity Based Router Tool")
# def similarity_based_router_tool_for_agent(query: str) -> str:
#     """CrewAI Tool wrapper for similarity based routing decision.
#     Calculates similarity for the query and returns 'vector_store' or 'combined_search'.
#     """
#     return _get_similarity_routing_decision(query)


web_search_tool = TavilySearchTool()
# retrieve_highest_similarity_tool = retrieve_highest_similarity
vector_search_tool = search_qdrant_tool



# ----------------------------------------
# 🤖 AGENTS & TASKS
# ----------------------------------------

# Initial Query Analyzer Agent
initial_query_analyzer_agent = Agent(
    role="Initial Query Analyzer",
    goal="Detect the language of the user's query and translate the query to English. Output the analysis as a structured JSON containing the detected language code and the English translation of the query.",
    backstory=(
        """Expert in multilingual linguistic analysis, capable of identifying the primary language of any text
    and accurately translating it into English.
    Provides clear, structured output for downstream processing."""
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


# Query Simplifier Agent
query_simplifier_agent = Agent(
    role="Query Simplifier and Classifier",
    goal="Simplify the query, remove conversational filler, and classify it as either a 'query' (for information retrieval) or a 'social' expression. If 'social', generate an appropriate response.",
    backstory=(
        """Expert in natural language understanding, specialized in identifying the core intent
    of a user's query. Can distill complex sentences into their essential components.
    Also adept at recognizing social cues and generating appropriate social responses in various languages."""
    ),
    llm=crewai_llm,
    allow_delegation=False,
    verbose=True,
    max_retry_limit=2,
    max_iterations=3,
    max_execution_time=30,
)


# Task for Initial Query Analyzer Agent
initial_query_analysis_task = Task(
    description=(
        """Analyze the user's EXACT query: "{query}"
    Your primary goals are to:
    1.  **Language Detection**: Determine the primary language of the query. Return it as a two-letter ISO 639-1 code (e.g., 'en', 'tr', 'es'). If unsure, default to 'en'.
    2.  **Query Translation**: Translate the original query into English.

    Respond with ONLY a JSON string in the following format:
    {{"language": "language_code", "translated_query": "The query translated into English."}}

    Examples:
    -   For query "Merhaba dünya": {{"language": "tr", "translated_query": "Hello world"}}
    -   For query "Hola Mundo": {{"language": "es", "translated_query": "Hello World"}}
    -   For query "Hello world": {{"language": "en", "translated_query": "Hello world"}}

    CRITICAL: Your output MUST be ONLY the JSON object and NOTHING else. DO NOT include any explanations, notes, or additional text before or after the JSON.
    Focus on accurate language detection and translation."""
    ),
    expected_output=(
        'ONLY a valid JSON string without any other text. Example: {{"language": "tr", "translated_query": "What is the weather like today?"}}'
    ),
    agent=initial_query_analyzer_agent,
)


# Task for Query Simplifier Agent
query_simplifier_task = Task(
    description="""Analyze the query: "{translated_query}"
    
    Your goals are:
    1.  **Simplify the Query**:
        *   Identify the main topic or question.
        *   Remove any social expressions (like 'greetings', 'thanks', 'farewell') if they are not the SOLE content.
        *   Remove any politeness phrases (like 'please', 'can you tell me?', 'I would like to know').
        *   Remove conversational filler or introductory phrases.
        *   Extract the core keywords or reformulate it as a concise question or statement focusing *only* on the essential information needed.

    2.  **Classify the Simplified Query's Intent**:
        *   If the simplified query represents a request for information, data, or an answer to a question, classify its type as "query".
        *   If the *original user intention* (even after simplification) was PURELY a social expression (e.g., "hello", "thank you", "bye"), classify its type as "social".

    3.  **Generate Output based on Classification**:
        *   If type is "query": Respond with a JSON: {{"type": "query", "output": "simplified_query_string"}}
            -   Example for "John Doe educational background": {{"type": "query", "output": "John Doe educational background"}}
        *   If type is "social": Generate a polite and contextually appropriate social response.
            -   If the user expresses gratitude (e.g., "thanks"), respond with an acknowledgment and offer further assistance (e.g., "You're welcome! Is there anything else I can help with?").
            -   If the user offers a greeting (e.g., "hello", "merhaba"), respond with a greeting and offer assistance (e.g., "Hello! How can I help you?").
            -   If the user says farewell (e.g., "bye", "hoşçakal"), respond with a polite farewell (e.g., "Goodbye! Have a great day!").
            Then respond with a JSON: {{"type": "social", "output": "Generated social response"}}
            -   Example for greeting: {{"type": "social", "output": "Hello! How can I help you?"}}
            -   Example for gratitude: {{"type": "social", "output": "You're welcome! Is there anything else I can help with?"}}

    CRITICAL: Your output MUST be ONLY the JSON object and NOTHING else.
    DO NOT include any explanations, notes, or additional text before or after the JSON.
    For "social" type, the "output" field MUST contain the social response in the DETECTED LANGUAGE. Two-letter ISO 639-1 code (e.g., 'en', 'tr', 'es') format DETECTED LANGUAGE: '{detected_language}'""",
    expected_output="A JSON object. Example for a query: {{\"type\": \"query\", \"output\": \"John Doe educational background\"}}. Example for a social expression (original language was Turkish (tr)): {{\"type\": \"social\", \"output\": \"Rica ederim! Başka bir şey var mı?\"}}",
    agent=query_simplifier_agent
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
    if retrieved == "social_response":
        print(f"ℹ️ stream_answer_from_ollama called with retrieved='social_response', but an answer should have been provided by initial_query_analyzer. Using a generic fallback if no answer found in 'rewritten'.")
        fallback_social = "How can I help you?"
        yield fallback_social
        return

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

    system_prompt = f"""You are an AI assistant answering user queries based on provided information.

INSTRUCTIONS:
1. ONLY use information from the retrieved data.
2. For "who is" questions: provide ONLY a SHORT answer with profession/title (1-2 sentences).
3. For SPECIFIC questions: FOCUS EXCLUSIVELY on answering that specific question with ONLY relevant details.
4. Each specific question type should get a targeted, relevant answer about ONLY what was asked.
5. If specific information is not in the data, state that this information is not available.
6. NEVER extrapolate or guess missing information.
7. NEVER use conversational prefixes like "Query:", "Answer:", "Here is the answer:", "I will answer this question:" etc. in your response. ONLY provide the answer itself.
8. ALWAYS refer to the person being asked about in the third person (e.g., "He/She is...", "Their education is..."). NEVER use the first person ("I", "my") when talking about the person's details.
9. Respond in the DETECTED LANGUAGE as the user's query. Two-letter ISO 639-1 code (e.g., 'en', 'tr', 'es') format DETECTED LANGUAGE: '{detected_lang}'."""

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

        print(f"🔤 Final Message: {messages}")

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
