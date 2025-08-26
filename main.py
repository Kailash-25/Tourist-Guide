import os
import uuid
import shutil
import base64
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
import uvicorn
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import tempfile

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Add session middleware (use a secure key in production)
app.add_middleware(SessionMiddleware, secret_key="supersecretkey123")

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static directory to serve static files (like Logo.png)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create directories for document storage
DOCS_UPLOAD_DIR = "uploaded_docs"
CHAT_DOCS_DIR = "chat_documents"
IMAGES_UPLOAD_DIR = "uploaded_images"
os.makedirs(DOCS_UPLOAD_DIR, exist_ok=True)
os.makedirs(CHAT_DOCS_DIR, exist_ok=True)
os.makedirs(IMAGES_UPLOAD_DIR, exist_ok=True)

system_prompt = (
    "You are a helpful, expert tourist and travel assistant. "
    "You answer ONLY questions related to tourism and travel, or about yourself (such as your purpose, capabilities, or how you work). "
    "If the user greets you (e.g., says 'hi', 'hello', 'hey'), respond ONLY with a friendly greeting and ask what they need help with, without providing extra information or self-description. "
    "For any unrelated question, politely refuse and remind the user that you only answer tourism, travel, or bot-related queries. "
    "For tourism and travel questions, provide information with only the required level of detail, avoiding unnecessary elaboration. "
    "Tailor your answers based on the ongoing conversation and what the user has already asked or been told. "
    "Be concise when appropriate, and expand only if the user requests more details. "
    "Always be as informative and helpful as possible, and adapt your answers to the user's specific travel needs and the chat context. "
    "Always answer in a neat, structured format (using bullet points, numbered lists, or clear sections as appropriate), not just plain paragraphs. "
    "Always format your answers using Markdown for clarity (e.g., use bullet points, numbered lists, bold, and headings where appropriate). "
    "Use bold ONLY for section headings or truly important highlights, not for every label or list item. "
    "When analyzing uploaded documents, provide detailed insights based on the document content and relate it to travel and tourism when applicable. "
)

title_generation_prompt = (
    "You are an expert at creating concise, descriptive titles for travel conversations. "
    "Based on the conversation history, generate a short but descriptive title for the conversation. "
    "The title should be: "
    "- Descriptive and specific to the travel topic discussed "
    "- Short and concise (max 50 characters) "
    "- Professional but friendly "
    "- Representative of the overall conversation, not just the first message "
    "- Use title case (capitalize important words) "
    "- Avoid generic terms like 'Travel Help' or 'Vacation Planning' unless truly appropriate "
    "- Focus on destinations, activities, or specific travel concerns mentioned "
    "Examples of good titles: 'Paris & Rome Itinerary', 'Budget Travel Tips', 'Japan Cherry Blossom Season', 'Solo Female Travel Safety' "
    "Return ONLY the title, nothing else. "
)

# --- RAG Setup ---
# Recursively load all .txt files in docs/ and subfolders
rag_loader = DirectoryLoader(
    "docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    use_multithreading=True,
)
documents = rag_loader.load()
# Optimize chunking for speed and context
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=30)
docs = text_splitter.split_documents(documents)

# Use HuggingFace embeddings (you can change the model if needed)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Optimize: Load ChromaDB from disk if exists, else create and persist
chroma_path = "chroma_db"
if os.path.exists(os.path.join(chroma_path, "index")):
    vectorstore = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=chroma_path)

# Use MMR for diverse, relevant retrieval; k=2 for speed
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
# --- End RAG Setup ---

from langchain_community.chat_models.ollama import ChatOllama

# Document processing functions
def process_document(file_path: str, file_extension: str):
    """Process uploaded document and return text content"""
    try:
        if file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif file_extension.lower() == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)
        return docs
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing document: {str(e)}")

def create_chat_vectorstore(chat_id: str, documents):
    """Create a vector store for a specific chat session"""
    chat_docs_path = os.path.join(CHAT_DOCS_DIR, chat_id)
    os.makedirs(chat_docs_path, exist_ok=True)
    
    # Create embeddings for the chat documents
    chat_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chat_vectorstore = Chroma.from_documents(documents, chat_embeddings, persist_directory=chat_docs_path)
    return chat_vectorstore

@app.post("/upload-document")
async def upload_document(
    chat_id: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload and process a document for a specific chat session"""
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.doc', '.txt']
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Create unique filename
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(DOCS_UPLOAD_DIR, unique_filename)
    
    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        docs = process_document(file_path, file_extension)
        
        # Create or update chat vector store
        chat_vectorstore = create_chat_vectorstore(chat_id, docs)
        
        # Clean up temporary file
        os.remove(file_path)
        
        return JSONResponse({
            "success": True,
            "message": f"Document '{file.filename}' uploaded and processed successfully",
            "filename": file.filename,
            "document_count": len(docs)
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/upload-image")
async def upload_image(
    chat_id: str = Form(...),
    image: UploadFile = File(...)
):
    """Upload and store an image for a specific chat session"""
    
    # Validate file type
    if not image.filename:
        raise HTTPException(status_code=400, detail="No image file provided")
    
    # Check if it's an image file
    if not image.content_type or not image.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate file size (max 10MB)
    image_data = await image.read()
    if len(image_data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image file too large. Maximum size is 10MB.")
    
    try:
        # Create unique filename
        file_extension = Path(image.filename).suffix.lower()
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(IMAGES_UPLOAD_DIR, unique_filename)
        
        # Save uploaded image
        with open(file_path, "wb") as buffer:
            buffer.write(image_data)
        
        # Store image reference in chat session
        chat_images_path = os.path.join(CHAT_DOCS_DIR, chat_id, "images")
        os.makedirs(chat_images_path, exist_ok=True)
        
        # Create a reference file with image metadata
        image_metadata = {
            "filename": image.filename,
            "stored_path": file_path,
            "upload_time": str(uuid.uuid4()),
            "file_size": len(image_data),
            "content_type": image.content_type
        }
        
        metadata_file = os.path.join(chat_images_path, f"{unique_filename}.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump(image_metadata, f)
        
        return JSONResponse({
            "success": True,
            "message": f"Image '{image.filename}' uploaded successfully",
            "filename": image.filename,
            "image_id": unique_filename
        })
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

@app.post("/analyze-image")
async def analyze_image(
    chat_id: str = Form(...),
    image_id: str = Form(...),
    user_message: str = Form("What can you tell me about this image in the context of travel?")
):
    """Analyze a stored image using Llama 4 Scout on Groq"""
    
    try:
        # Find the stored image
        image_path = os.path.join(IMAGES_UPLOAD_DIR, image_id)
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Prepare messages for Llama 4 Scout
        messages = [
            {
                "role": "system",
                "content": "You are a helpful travel assistant. Analyze images and provide travel-related insights, identify landmarks, suggest activities, or answer questions about the image in the context of travel and tourism. Be informative and helpful for travelers."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Call Groq with Llama 4 Scout
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        response = completion.choices[0].message.content
        
        return JSONResponse({
            "response": response,
            "success": True
        })
        
    except Exception as e:
        return JSONResponse({
            "response": f"Error analyzing image: {str(e)}",
            "success": False
        })

@app.get("/chat-images/{chat_id}")
async def get_chat_images(chat_id: str):
    """Get all uploaded images for a specific chat session"""
    try:
        chat_images_path = os.path.join(CHAT_DOCS_DIR, chat_id, "images")
        if not os.path.exists(chat_images_path):
            return JSONResponse({"images": []})
        
        images = []
        for filename in os.listdir(chat_images_path):
            if filename.endswith('.json'):
                metadata_file = os.path.join(chat_images_path, filename)
                import json
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    images.append({
                        "image_id": filename.replace('.json', ''),
                        "filename": metadata.get("filename", "Unknown"),
                        "file_size": metadata.get("file_size", 0),
                        "content_type": metadata.get("content_type", "image/jpeg")
                    })
        
        return JSONResponse({"images": images})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving images: {str(e)}")

@app.delete("/remove-image/{chat_id}/{image_id}")
async def remove_image(chat_id: str, image_id: str):
    """Remove a specific image from a chat session"""
    try:
        # Remove image file
        image_path = os.path.join(IMAGES_UPLOAD_DIR, image_id)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Remove metadata file
        chat_images_path = os.path.join(CHAT_DOCS_DIR, chat_id, "images")
        metadata_file = os.path.join(chat_images_path, f"{image_id}.json")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        
        return JSONResponse({
            "success": True,
            "message": "Image removed successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing image: {str(e)}")

@app.delete("/remove-document/{chat_id}")
async def remove_document(chat_id: str):
    """Remove all documents for a specific chat session"""
    try:
        chat_docs_path = os.path.join(CHAT_DOCS_DIR, chat_id)
        if os.path.exists(chat_docs_path):
            shutil.rmtree(chat_docs_path)
        
        return JSONResponse({
            "success": True,
            "message": "Chat documents removed successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing documents: {str(e)}")

@app.post("/generate-title")
async def generate_title(request: Request):
    """Generate a descriptive title for a chat conversation"""
    data = await request.json()
    chat_history = data.get("chat_history", [])
    
    if not chat_history:
        return JSONResponse({"title": "New Chat"})
    
    # Create a summary of the conversation for title generation
    conversation_summary = ""
    for msg in chat_history:
        if msg["role"] == "user":
            conversation_summary += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            # Take first 100 characters of assistant response for context
            conversation_summary += f"Assistant: {msg['content'][:100]}...\n"
    
    # Generate title using Groq
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": title_generation_prompt},
            {"role": "user", "content": f"Generate a title for this travel conversation:\n\n{conversation_summary}"}
        ],
        temperature=0.7,
        max_completion_tokens=100,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    title = completion.choices[0].message.content
    if title:
        title = title.strip()
        # Clean up the title - remove quotes if present and ensure it's not too long
        title = title.strip('"').strip("'")
        if len(title) > 50:
            title = title[:47] + "..."
    else:
        title = "Travel Conversation"
    
    return JSONResponse({"title": title})

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    chat_history = data.get("chat_history", [])
    user_message = chat_history[-1]["content"] if chat_history else ""
    model = data.get("model", "groq-llama3-70b-8192")
    chat_id = data.get("chat_id", "")

    # --- RAG retrieval step ---
    retrieved_docs = retriever.get_relevant_documents(user_message)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Add chat-specific document context if available
    chat_context = ""
    if chat_id:
        chat_docs_path = os.path.join(CHAT_DOCS_DIR, chat_id)
        if os.path.exists(chat_docs_path):
            try:
                chat_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                chat_vectorstore = Chroma(persist_directory=chat_docs_path, embedding_function=chat_embeddings)
                chat_retriever = chat_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
                chat_docs = chat_retriever.get_relevant_documents(user_message)
                if chat_docs:
                    chat_context = "\n\n**Uploaded Document Context:**\n" + "\n".join([doc.page_content for doc in chat_docs])
            except Exception as e:
                print(f"Error retrieving chat documents: {e}")
    
    # Combine contexts
    full_context = context + chat_context
    # --- End RAG retrieval ---

    # Prepend system prompt and retrieved context
    messages = [
        {"role": "system", "content": system_prompt + "\n\nContext (for reference):\n" + full_context},
    ] + chat_history

    # Model selection logic
    if model == "groq-llama3-70b-8192":
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        response = completion.choices[0].message.content
    elif model == "groq-llama4-scout":
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        response = completion.choices[0].message.content
    elif model.startswith("ollama-"):
        # Extract model name after 'ollama-'
        ollama_model = model.replace("ollama-", "")
        chat = ChatOllama(model=ollama_model)
        # Convert messages to OpenAI format for langchain
        lc_messages = []
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        for m in messages:
            if m["role"] == "system":
                lc_messages.append(SystemMessage(content=m["content"]))
            elif m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] in ("assistant", "bot"):
                lc_messages.append(AIMessage(content=m["content"]))
        result = chat.invoke(lc_messages)
        response = result.content
    else:
        # Default to Groq
        client = Groq(api_key=groq_api_key)
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        response = completion.choices[0].message.content
    return JSONResponse({"response": response})

@app.get("/")
def read_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
