import streamlit as st
import time
from main import (
    load_pdf,
    chunk_text,
    save_to_qdrant,
    stream_answer_from_ollama,
    qdrant_client,
    Distance,
    VectorParams,
    process_query,
)

st.set_page_config(page_title="Person Information Chatbot", layout="wide")

# Retrieve Available Qdrant Collections And Store Them In Session State
if "collections" not in st.session_state:
    try:
        existing_collections = qdrant_client.get_collections().collections
        st.session_state.collections = [c.name for c in existing_collections]
    except Exception as e:
        st.error(f"Failed to retrieve collections: {e}")
        st.session_state.collections = []

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None


# Sidebar Panel For General Settings
with st.sidebar:
    st.title("⚙️ Settings & Information")
    query_time_placeholder = st.empty()
    st.markdown("---")

    # Manage And Select Qdrant Vector Collections
    st.title("🧠 Qdrant Collections")

    if "collections" not in st.session_state:
        st.session_state.collections = []
    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None

    new_collection_name = st.text_input(
        "New Collection Name", placeholder="example: cv_pdfs"
    )

    if st.button("📁 Create Collection"):
        if (
            new_collection_name
            and new_collection_name not in st.session_state.collections
        ):
            try:
                qdrant_client.create_collection(
                    collection_name=new_collection_name,
                    # 1024 size is the default size for the embedding model we are using. (bge-m3)
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
                )
                if new_collection_name not in st.session_state.collections:
                    st.session_state.collections.append(new_collection_name)
                st.session_state.collection_name = new_collection_name
                st.success(
                    f"✅ '{new_collection_name}' collection has been created and selected."
                )
            except Exception as e:
                if "already exists" in str(e):
                    st.warning(
                        f"⚠️ '{new_collection_name}' collection already exists. You can select it from below."
                    )
                else:
                    st.error(f"Error: {e}")

    for name in st.session_state.collections:
        if st.button(name, key=name):
            st.session_state.collection_name = name

    if st.session_state.collection_name:
        st.sidebar.success(f"🔗 Active Collection: {st.session_state.collection_name}")
    else:
        st.sidebar.warning("❗No active collection has been selected yet.")

    # Upload Pdf Files And Index Them Into Qdrant As Vector Chunks
    st.title("📎 PDF Upload and Indexing")
    uploaded_pdfs = st.file_uploader(
        "Upload your PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_pdfs:
        if st.button("📚 Chunk & Index"):
            all_chunks = []
            for uploaded_file in uploaded_pdfs:
                pdf_name = uploaded_file.name
                st.write(f"🔍 `{pdf_name}` processing...")
                full_text = load_pdf(uploaded_file)
                chunks = chunk_text(full_text)
                all_chunks.extend(chunks)
            save_to_qdrant(all_chunks)
            st.success(
                f"✅ Total {len(all_chunks)} amounts of chunks uploaded to Qdrant."
            )


# Load And Display Previous Chat Messages From Session State To Maintain Conversation History
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Process User Input Through a Multi-Step Agent Pipeline
st.title("💬 Person Information Query System")
user_input = st.chat_input("Who would you like to know about?...")

if user_input:
    print(f"👤 User Query: {user_input}")
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    left_col, right_col = st.columns([3, 1])
    with left_col:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            query_time_placeholder = st.empty()
            start_time = time.time()

            try:
                # Process the query using main processing function
                status_placeholder.info("🔄 Processing started...")
                result = process_query(user_input)

                # Get the results
                routing = result["routing"]
                rewritten = result["rewritten"]
                retrieved = result["retrieved"]
                evaluation = result["evaluation"]
                error = result["error"]

                # Error handling
                if error:
                    status_placeholder.error(f"🚨 Error occurred: {error}")
                    st.error(f"An error occurred: {error}")
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Sorry, an error occurred while processing your query: {error}",
                        }
                    )
                # Check if the query was a social expression and handled directly by process_query
                elif routing == "social" and "answer" in result and result["answer"]:
                    status_placeholder.success("✅ Answer ready!")
                    social_answer = result["answer"]
                    st.markdown(social_answer)
                    print(f"💬 Social Expression Answer: {social_answer}")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": social_answer}
                    )
                else:
                    # Answer step (streaming) for non-social expression queries
                    status_placeholder.info("💬 Streaming answer...")
                    stream_area = st.empty()
                    streamed_text = ""
                    first_token_received = False

                    try:
                        for token in stream_answer_from_ollama(
                            rewritten, retrieved, evaluation, result["language"]
                        ):
                            if not first_token_received:
                                first_token_received = True
                                first_token_time = time.time()
                                query_time_placeholder.write(
                                    f"⏳ **First Token Latency:** {first_token_time - start_time:.2f} seconds"
                                )
                            streamed_text += token
                            stream_area.markdown(streamed_text + "▌")

                        stream_area.markdown(streamed_text)
                        status_placeholder.success("✅ Answer ready!")
                        print(f"💬 Final Answer: {streamed_text}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": streamed_text}
                        )
                    except Exception as e:
                        status_placeholder.error(
                            f"🚨 Error generating answer: {str(e)}"
                        )
                        final_error_message = f"I'm sorry, but I encountered an error while generating your answer. Error details: {str(e)}"
                        print(f"❌ Error during answer generation: {str(e)}")
                        stream_area.markdown(final_error_message)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_error_message}
                        )

                # Calculate and display query time
                end_time = time.time()
                query_time = end_time - start_time
                query_time_placeholder.info(f"⏱️ Query Time: {query_time:.2f} seconds")

            except Exception as e:
                status_placeholder.error(f"🚨 Unexpected error: {str(e)}")
                print(f"Error in main flow: {str(e)}")
                st.error(f"An error occurred: {str(e)}")
                st.write(
                    "An error occurred while processing your query. Please try again."
                )

    with right_col:
        with st.expander("🧠 Agent Insights", expanded=True):
            st.markdown(
                f"📝 **Rewritten Query:**\n\n`{result['rewritten'] if 'result' in locals() and result.get('rewritten') else user_input}`"
            )
            st.markdown(
                f"📍 **Routing Decision:** `{result['routing'] if 'result' in locals() and result.get('routing') else 'N/A'}`"
            )
            st.markdown("📦 **Retrieved Information:**")
            st.markdown(
                f"```text\n{result['retrieved'] if 'result' in locals() and result.get('retrieved') else 'N/A'}\n```"
            )
            st.markdown(
                f"🧠 **Evaluation Decision:** `{result['evaluation'] if 'result' in locals() and result.get('evaluation') else 'N/A'}`"
            )
