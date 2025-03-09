# 🧠 Smart Content Chatbot

Welcome to the **Smart Content Chatbot** repository! 🚀 This project is a **Streamlit-based chatbot** that processes content from YouTube videos and webpages, transforming it into a conversational interface powered by **advanced AI models**. The chatbot utilizes **LangChain, FAISS, and HuggingFace** to create an intelligent assistant capable of answering questions based on processed content.

---

## ✨ Features

✅ **Content Loading**: Automatically fetch content from YouTube videos or webpages.  
✅ **Text Processing & Splitting**: Breaks large text into manageable chunks.  
✅ **Vector Embeddings & Storage**: Uses FAISS and HuggingFace embeddings for efficient retrieval.  
✅ **Conversational Memory**: Maintains chat history for context-aware responses.  
✅ **AI-Powered Chat**: Uses LangChain to generate smart, relevant responses.  
✅ **Interactive UI**: Built with **Streamlit** for a seamless user experience.

---

## 📂 Repository Structure

```
📦 Smart-Content-Chatbot
├── 📄 env.examples     # Example environment file with API keys
├── .streamlit/
│   └── secrets.toml
├── 📝 process.py       # Backend logic: content fetching, vectorization, and chat logic
├── 🎭 chatbot_app.py    # Streamlit UI for chatbot interaction
└── 📜 README.md        # Project documentation
```

---

## 🔧 Prerequisites

Ensure you have **Python 3.8+** installed. Install dependencies using:

```bash
pip install streamlit python-dotenv langchain langchain_community langchain_huggingface langchain_groq
```

> ⚠️ **Note:** Some libraries may have additional dependencies. Check their documentation for details.

---

## 🚀 Setup Instructions

1️⃣ **Clone the Repository**

```bash
git clone <repository_url>
cd <repository_directory>
```

2️⃣ **Configure Environment Variables**

```bash
cp env.examples .env
nano .env  # Edit and add your API keys
```

```ini
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

3️⃣ **Run the Chatbot**

```bash
streamlit run chatbot_app.py
```

🌟 The chatbot UI will open in your browser. Enter a YouTube or webpage URL, and start chatting! 💬

---

## 🔁 Flow Diagram

```plaintext
[ User Input (YouTube/Webpage URL) ]
              ↓
     [ Streamlit UI ]
              ↓
  [ Content Processing (process.py) ]
              ↓
 [ Text Splitting & Embeddings ]
              ↓
    [ FAISS Vector Store ]
              ↓
    [ Query Processing ]
              ↓
 [ Conversational Memory ]
              ↓
 [ AI Response Generation ]
              ↓
    [ Display in Streamlit UI ]
```

---

## 🛠 How It Works

🔹 **Content Fetching:** Uses `YoutubeLoader` or `WebBaseLoader` from `langchain_community.document_loaders`.  
🔹 **Text Processing & Embeddings:** Splits text with `RecursiveCharacterTextSplitter` and embeds it using `HuggingFaceEmbeddings`.  
🔹 **Vector Storage:** Stores embeddings in a FAISS vector database for quick retrieval.  
🔹 **AI-Powered Chat:** Uses LangChain to generate responses with conversation memory.  
🔹 **Interactive UI:** Streamlit provides an intuitive interface for user interactions.  

---

## 🤝 Contributing

💡 **Ideas for Improvement?** Open an issue or submit a pull request! Contributions are welcome. Follow standard GitHub guidelines. 🙌

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

## 📬 Contact

For questions or support, reach out via [GitHub Issues](https://github.com/your_username/your_repo/issues) or email **mohamed42468@gmail.com**.

---

### 📚 References

📌 **LangChain Docs:** [LangChain Documentation](https://docs.langchain.com)  
📌 **FAISS:** [FAISS GitHub Repository](https://github.com/facebookresearch/faiss)  
📌 **Streamlit Docs:** [Streamlit Documentation](https://docs.streamlit.io)  

🚀 **Happy Coding & Enjoy Building Your AI Chatbot!** 🤖🎉

