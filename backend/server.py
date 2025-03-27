from flask import Flask, request, jsonify
import os
import pinecone
from langchain_community.vectorstores import Pinecone as PineconeVector
from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.memory import ConversationBufferMemory
from llama_cpp import Llama
from flask_cors import CORS
import PyPDF2
import logging
import uuid
import re



"""


This script initializes a Flask server to handle chatbot interactions. 
It processes user queries, retrieves relevant context from Pinecone, and 
generates responses using the Llama 3 8B model.

     Features:
- Accepts user input and character selection
- Retrieves context from Pinecone for memory management
- Generates dynamic responses using Llama 3 8B
- Implements emotion simulation using Psi Theory(adjustable parameters by the user based on preference) 
- By dynamically updating emotional states



Langchain is used to managing the embeddings and vector storage, 
which play a crucial role in enhancing the chatbot's ability to understand and retrieve context. 
Specifically, LangChain's PineconeVector and OpenAIEmbeddings are used to handle vector storage 
and manage the interaction between the Pinecone index and the Llama model for memory management.


"""


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if the index exists, and create it if it doesn't
index_name = "chatbot-memory"

# Initialize Pinecone

try:
    pc = Pinecone(api_key="pinecone-api")
    pc.describe_index(index_name)  # Verify connection
except Exception as e:
    print(f"Pinecone initialization failed: {e}")
    raise



if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust dimension based on your embeddings
        metric="cosine",  # Adjust metric as needed
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Adjust region as needed
        )
    )

# Load LLM model
model = Llama(
    
                model_path="ai_simulator/backend/Meta-Llama-3.1-8B-GGUF/Meta-Llama-3-8B-Q4_K_M.gguf", 
                n_gpu_layers=5,
                n_threads=4,  # Adjust based on your CPU cores
                n_batch=256,   # Reduce if you get out-of-memory issues
                verbose=False
              )

# Verify model loading
if not model:
    raise RuntimeError("Model failed to load - check model path")

# Function to extract text from uploaded books
def extract_text_chunks(file_path, chunk_size=300):
    if file_path.endswith('.pdf'):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
        
        # Split into manageable chunks
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    else:
        return []


def clean_character_list(text):
    # Remove special tokens and artifacts
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\(.*?\)', '', text)  # Remove anything in parentheses
    
    # Extract names with proper capitalization
    names = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Remove numbering/bullets
        line = re.sub(r'^[\d\.\-\*]+', '', line).strip()
        
        # Extract just the name part
        name = re.split(r'[,;:]', line)[0].strip()
        name = re.sub(r'\b(?:Mr|Mrs|Ms|Dr)\.?\s+', '', name)  # Remove titles
        
        if 2 <= len(name) <= 30 and name[0].isupper():
            names.append(name)
    
    # Remove duplicates while preserving order
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]

def extract_characters_from_chunks(chunks):
    """Process text chunks to extract character names"""
    all_characters = set()
    
    for chunk in chunks:
        try:
            # Properly formatted prompt for Llama 3
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert literary analyst. Extract ONLY proper names of characters from the text.
Return JUST the names, one per line, with no additional commentary.
<|start_header_id|>user<|end_header_id|>
Extract character names from this text:
{chunk[:3000]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Character names:"""
            
            response = model(
                prompt,
                max_tokens=500,
                temperature=0.1,
                stop=["<|eot_id|>", "\n\n"]
            )
            
            if response and 'choices' in response and len(response['choices']) > 0:
                characters = clean_character_list(response['choices'][0]['text'])
                all_characters.update(characters)
                
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            continue
    
    return sorted(all_characters)

# Endpoint to upload books and extract characters
@app.route('/upload', methods=['POST'])
def upload_book():
    """Endpoint to upload books and extract characters"""
    print("Received request:", request)
    print("Request files:", request.files)  # Debugging

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    try:
        file = request.files['file']
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400
        
        os.makedirs("uploads", exist_ok=True)
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        chunks = extract_text_chunks(file_path)
        if not chunks:
            return jsonify({"error": "No text could be extracted"}), 400
        
        # Process first 3 chunks to limit processing time
        characters = extract_characters_from_chunks(chunks[:3])
        
        if not characters:
            return jsonify({"error": "No characters could be identified"}), 404
            
        return jsonify({
            "characters": characters,
            "count": len(characters),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

def clean_bot_response(text):
    tokens_to_remove = ["<<SYS>>", "</SYS>>", "[/INST]", "<|eot_id|>"]
    for token in tokens_to_remove:
        text = text.replace(token, "")
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


"""
    Handles user input and generates AI responses.

    Request Body:
    - `message` (str): User's input text.
    - `character` (str): Selected character for the conversation.
    
    Returns:
    - `response` (str): AI-generated reply based on character personality & memory.
    """

# Initialize LangChain's memory for storing the conversation history
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Function to create context from memory
def create_context():
    return memory.buffer

# Endpoint to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    print(f"\n=== NEW REQUEST ===")  # Should appear ONCE per message
    print(f"Input: {request.json.get('message')}")
    print(f"Received JSON: {request.json}")  
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        user_input = data.get("message", "").strip()
        character = data.get("character", "").strip()
        
        if not user_input:
            return jsonify({"error": "Message cannot be empty"}), 400
        if not character:
            return jsonify({"error": "Character not specified"}), 400
        
        # Add user input to memory as part of conversation context
        memory.add_user_message(user_input)

        # Build the prompt from the memory
        context = create_context()

        # Initialize Pinecone index (if used)
        try:
            index = pc.Index(index_name)
            index.describe_index_stats()  # Test connection
        except Exception as e:
            print(f"Pinecone warning: {str(e)}")

        # Get conversation context (if available)
        context = ""
        try:
            embedding = [0.1] * 1536  # Dummy embedding if OpenAI fails
            results = index.query(vector=embedding, top_k=2, include_metadata=True)
            context = "\n".join([res["metadata"]["text"] for res in results["matches"]])
        except Exception as e:
            print(f"Context retrieval warning: {str(e)}")


        # Retrieve past conversation context
        past_messages = memory.load_memory_variables({})["history"]

        prompt = f"""
        [INST] <<SYS>>
        You are {character}, a character from literature. Respond naturally while staying in character. 
        Maintain context from past messages.
        <</SYS>>
        
        Past Conversation: {past_messages}

        User Input: {user_input}
        [/INST]
        """

        print(f"Sending prompt to model:\n{prompt}")  # Debug logging

        # Get model response with stricter generation settings
        try:
            response = model(
                prompt,
                max_tokens=20,  # Reduced to prevent long repetitions
                temperature=0.7,
                top_p=0.9,
                echo=False,
                stop=[ "[/INST]"]  # More stop conditions( "<|end_of_text|>", "\n\n", "<<SYS>>""<|eot_id|>" ,)
            )
            
            if not response or 'choices' not in response:
                raise ValueError("Empty model response")
            
            bot_response = response["choices"][0]["text"].strip() if "choices" in response else response["text"].strip()

            print(f"Cleaned response: {bot_response}")  # Debug logging
            
            if not bot_response:
                raise ValueError("Empty bot response")
            
            # Update emotional states using the global 'emotional_states'
            global emotional_states  # Ensure you're modifying the global variable
            emotional_states = update_emotions(emotional_states, user_input, bot_response)

            # Add bot's response to memory
            memory.add_ai_message(bot_response)
        
            # Store conversation 
            try:
                index.upsert([(
                    str(uuid.uuid4()),
                    embedding,
                    {"text": bot_response, "character": character}
                )])
            except Exception as e:
                print(f"Storage warning: {str(e)}")

            return jsonify({"response": bot_response, "status": "success"})
            
        except Exception as e:
            print(f"Model error: {str(e)}")
            return jsonify({"response": f"{character} is thinking...", "status": "warning"})

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/get_parameters', methods=['GET'])
def get_parameters():
    character = request.args.get("character")
    # Fetch parameters for the character (replace with your logic)
    params = {
        "Valence Level": 50,
        "Arousal Level": 50,
        "Selection Threshold": 50,
        "Resolution Level": 50,
        "Goal-Directedness": 50,
        "Securing Rate": 50
    }
    return jsonify({"psychological_params": params})

@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    character = request.args.get("character")
    
    if not character:
        return jsonify({"error": "Character not specified"}), 400

    # Reset emotional states when a character is selected
    emotional_states = {
        "Anger": 0,
        "Sadness": 0,
        "Joy": 0,
        "Pride": 0,
        "Bliss": 0
    }
    
    return jsonify({"emotional_states": emotional_states})



# Psychological parameters and emotional states
psychological_params = {
    "Valence Level": 0.5,
    "Arousal Level": 0.5,
    "Selection Threshold": 0.5,
    "Resolution Level": 0.5,
    "Goal-Directedness": 0.5,
    "Securing Rate": 0.5
}

emotional_states = {
    "Anger": 0.0,
    "Sadness": 0.0,
    "Joy": 0.0,
    "Pride": 0.0,
    "Bliss": 0.0
}

def update_emotions(emotional_states, user_input, bot_response):
    """
    Update the emotional states based on the user's input and the bot's response.
    This is a very basic example that adjusts the emotions based on keyword presence.
    """
    # Keywords for basic emotion detection
    keywords = {
        "anger": ["angry", "rage", "frustration"],
        "sadness": ["sad", "cry", "upset", "depressed"],
        "joy": ["happy", "joy", "excited", "laugh"],
        "pride": ["proud", "accomplished", "success"],
        "bliss": ["bliss", "peace", "content"]
    }
    
    # Simple emotion update based on user input
    for emotion, words in keywords.items():
        if any(word in user_input.lower() for word in words) or any(word in bot_response.lower() for word in words):
            emotional_states[emotion] += 0.1  # Increase the emotion by a small amount
            emotional_states[emotion] = min(emotional_states[emotion], 1.0)  # Ensure it doesn't exceed 1.0
    
    # Optional: Add a decay effect to gradually reduce emotions over time
    for emotion in emotional_states:
        emotional_states[emotion] *= 0.99  # Decay effect (emotions gradually fade)
    
    return emotional_states


# debugging


# @app.route('/health')
# def health_check():
#     return jsonify({"status": "healthy"})




# def fetch_memory(query):
#     """
#     Retrieves relevant memory context from Pinecone.

#     Parameters:
#     - `query` (str): User's latest input.

#     Returns:
#     - `context` (str): Retrieved memory for better continuity.
#     """
#     results = index.query(vector=embed_text(query), top_k=2, include_metadata=True)
#     return "\n".join([res["metadata"]["text"] for res in results["matches"]]) if results else ""


# def embed_text(text):
#     """Converts text into embeddings (mock implementation)."""
#     return [0.1, 0.2, 0.3]  # Replace with real embedding model

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # Different port
