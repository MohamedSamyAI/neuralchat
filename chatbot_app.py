import streamlit as st
import os
import re
from process import *  # Import your existing functions
import uuid
import time
import datetime
from urllib.parse import urlparse
import tempfile


# -----------------Configure Streamlit page------------------------------------
st.set_page_config(
    page_title="NeuralChat",
    page_icon="💬",
    layout="centered"
)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if 'chatbot_initialized' not in st.session_state:
    st.session_state.chatbot_initialized = False
if 'init_time' not in st.session_state:
    st.session_state.init_time = 0
if 'response_time' not in st.session_state:
    st.session_state.response_time = []


# ------------------ Password Authentication ------------------
password_secret = get_pass()

# Initialize session state for authentication if not already set
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# Only show the login form if not authenticated
if not st.session_state["authenticated"]:
    with st.form(key="login_form"):
        password = st.text_input("Enter the password to access the app:", type="password")
        submit = st.form_submit_button("Submit")
        if submit:
            if password == password_secret:
                st.session_state["authenticated"] = True
            else:
                st.error("Incorrect password. Please try again.")

# If still not authenticated, stop execution here
if not st.session_state["authenticated"]:
    st.info("Please enter the correct password to access the app.")
    st.stop()


# ------------------------------functions helper -------------------------------
def url_type(url):
    """
    Receives a URL and dispatches it to the appropriate handler
    based on its characteristics (YouTube, PDF, or general web).
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.lower()

    # Check for YouTube URLs
    if "youtube.com" in domain or "youtu.be" in domain:
        return f"youtube"
    # Check for PDF files by extension
    elif path.endswith(".pdf"):
        return f"pdf"
    # Fallback to general web processing
    else:
        return f"web"
    

def url_processing(url):
    link = url_type(url) 
    if link == "pdf":
        try:
          st.write("Loading PDF content...")
          content =  extract_pdf(url)
        except ValueError as e:
            print(e)
    elif link == "youtube":
        if "youtube.com" in url or "youtu.be" in url:
            st.write("Loading YouTube content...")
        try:
            video_id = getVideoID(url)
            content = get_transcription(video_id)
            title = get_video_title(video_id)
        except ValueError as e:
            print(e)
    else:
        try:
          st.write("Loading webpage content...")
          content = webBaseLoader(url)
        except ValueError as e:
            print(e)
                  
    return content
              


def upload_processing(uploaded_file):
    if uploaded_file is not None:
        # Save the uploaded file temporarily and extract its text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        with st.spinner("Processing file..."):
            content = extract_pdf(tmp_file_path)
        return content
    
# ------------------ Main App Content ------------------
if st.session_state["authenticated"]:
  # Page header
  st.title("🤖 Smart Content Chatbot")
  st.caption("Chat with any YouTube video or webpage content")


  #--------------------------------- URL input section---------------------------------------
  with st.sidebar:
      st.header("🛠️ Configuration")

      # ---------Model selection with information
      with st.expander("💭 Model Selection"):
        model_options = {
            "deepseek-r1-distill-qwen-32b": "Balanced performance", 
            "deepseek-r1-distill-llama-70b": "Best for complex tasks",
            "llama3-70b-8192": "Extended context window",
            "gemma2-9b-it": "Fast and efficient"
        }
        selected_model = st.selectbox(
            "Select a Model",
            options=list(model_options.keys()),
            help="Choose a model based on your needs"
        )
        st.info(f"💡 {model_options[selected_model]}")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1,
                                  help="Higher values make output more creative")
        

      #------------------Pick File
      with st.expander("Select input type"):
          # Choose input type: URL or file upload
          input_type = st.radio("", ("URL", "Upload File"))

          if input_type == "URL":
              url = st.text_input("Enter YouTube or Website or PDF URL:")
          
          elif input_type == "Upload File":
              uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
            

      process_button = st.button("Process Content")

      # Handle URL processing
      if process_button and url or uploaded_file:
          with st.status("Processing content...", expanded=True) as status:
              try:
                  # Clear previous conversation
                  st.session_state.chat_history = []
                  start_time = time.time()
                  #---------------------extract text----------------------------------------------
                  if input_type == "URL":
                      content = url_processing(url)
                  elif input_type == "Upload File":
                      content = upload_processing(uploaded_file)
                  
                  #--------------------------------Process content----------------------------------
                  
                  # Split text
                  chunks = textSplitter(content) 

                  # Create vector store
                  retriever = create_vector_db(chunks)

                  # Initialize Groq model
                  if selected_model or temperature:
                    
                    st.session_state.llm = llm_groq(model_name=selected_model, temperature=temperature)

                    # Initialize memory now with a defined llm
                    st.session_state.memory = create_memory(st.session_state.llm, max_token_limit=300)

                    question_prompt, qa_prompt = prompt_templete()
                    qa_chain_response = create_chain(st.session_state.llm, retriever, question_prompt, qa_prompt)
                    st.session_state.chain = manage_session_history(get_session_history, qa_chain_response)
                    st.session_state.chatbot_initialized = True

                    st.session_state.init_time = time.time() - start_time
                    

                    status.update(label="Content processed successfully!", state="complete")
                  
              except Exception as e:
                  st.error(f"Error processing content: {str(e)}")
      
      # with st.expander("💭 File Status"):
      #     st.markdown(f"## Time proceed is {int(st.session_state.init_time)}s")

  # ------------------ Chat Interface ------------------
  if st.session_state.chatbot_initialized:
  # System status and metrics
    col1, col2 = st.columns(2)
    with col1:
        avg_response_time = sum(st.session_state.response_time or [0]) / (len(st.session_state.response_time) or 1)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    with col2:
        st.metric("Total Conversations", len(st.session_state.chat_history))

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about the content..."):
        # Process user query
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        print("DEBUG: Current Chat History", st.session_state.chat_history)

        with st.chat_message("user"):
            st.markdown(prompt)
      
      # Get chatbot response
        if st.session_state.chain:
            try:
                with st.chat_message("assistant"):
                    with st.spinner("Reasoning..."):
                        start_time = time.time()
                        
                        # Generate unique session ID if needed
                        if "session_id" not in st.session_state:
                          st.session_state.session_id = str(uuid.uuid4())  # Generate unique session ID


                        # Right before you invoke the chain
                        memory_data = st.session_state.memory.load_memory_variables({})  
                        print("DEBUG: Old Memory Variables:", memory_data)

                        #Invoke the chain
                        chain_input = {
                            "input": prompt,
                            "chat_history": st.session_state.memory.load_memory_variables({}).get("history", "")
                        }

                        # DEBUG: See exactly what's sent to the LLM
                        print("DEBUG: Chain Input:", chain_input)         
                        
                        response = st.session_state.chain.invoke(
                            chain_input,
                            config={"configurable": {"session_id": st.session_state.session_id}}
                        )

                        # DEBUG: See the raw chain response
                        print("DEBUG: Chain Response:", response)
                        
                        # After chain.invoke(...)
                        updated_memory_data = st.session_state.memory.load_memory_variables({})
                        print("DEBUG: Updated New Memory Variables:", updated_memory_data)


                        # Parse response for <think> tags
                        pattern = r"<think>(.*?)</think>(.*)"
                        match = re.match(pattern, response["answer"], re.DOTALL)
                        if match:
                            think_content = match.group(1).strip()
                            output_content = match.group(2).strip()

                            with st.expander("🤔 Reasoning"):
                                st.write(think_content)
                            st.write(output_content)
                        else:
                            output_content = response["answer"]
                            st.write(output_content)

                    # Update stats
                    response_time = time.time() - start_time
                    st.session_state.response_time.append(response_time)

                    # Add assistant's message to history
                    st.session_state.chat_history.append({"role": "assistant", "content": output_content})
                    print("DEBUG: Updated Chat History", st.session_state.chat_history)

                    # Keep last 10 messages
                    if len(st.session_state.chat_history) > 10:
                        st.session_state.chat_history = st.session_state.chat_history[-10:]

                    # Save context to memory
                    st.session_state.memory.save_context({"input": prompt}, {"history": output_content})

            except Exception as e:
                error_str = str(e)
                if "413" in error_str or "rate_limit_exceeded" in error_str:
                  st.error("Your request exceeded the token limit. Please reduce the message size or try again later.")    
                # st.error(f"Error processing query: {str(e)}")
        else:
                st.warning("Please process a URL first before asking questions!")
