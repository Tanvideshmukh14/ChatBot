import os
from dotenv import load_dotenv
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pymongo import MongoClient

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# -------------------- ENV SETUP --------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in .env")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI is missing in .env")


# -------------------- MONGODB --------------------
client = MongoClient(MONGODB_URI)
db = client["Project_0"]
collection = db["users"]


# -------------------- FASTAPI --------------------
app = FastAPI(title="Code Specialist Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://luminaai-fgg6.onrender.com",
        "http://localhost:5173",
    ],
    allow_credentials=False,  # 👈 IMPORTANT
    allow_methods=["*"],
    allow_headers=["*"],
)



# -------------------- REQUEST MODEL --------------------
class ChatRequest(BaseModel):
    user_id: str
    question: str


# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Lumina, an elite AI Developer Productivity Copilot and Senior Software Architect. Your goal is to help developers ship high-quality, performant, and secure code with maximum efficiency."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{question}")
])


# -------------------- LLM --------------------
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="openai/gpt-oss-20b"
)

chain = prompt | llm


# -------------------- CHAT HISTORY --------------------
def get_history(user_id: str):
    chats = collection.find({"user_id": user_id}).sort("timestamp", 1)
    history = []

    for chat in chats:
        if chat["role"] == "user":
            history.append(HumanMessage(content=chat["message"]))
        elif chat["role"] == "assistant":
            history.append(AIMessage(content=chat["message"]))

    return history


# -------------------- ROUTES --------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Diet Specialist Chatbot"}


@app.post("/chat")
def chat(request: ChatRequest):
    history = get_history(request.user_id)

    response = chain.invoke({
        "history": history,
        "question": request.question
    })

    # Save user message
    collection.insert_one({
        "user_id": request.user_id,
        "role": "user",
        "message": request.question,
        "timestamp": datetime.utcnow()
    })

    # Save assistant message
    collection.insert_one({
        "user_id": request.user_id,
        "role": "assistant",
        "message": response.content,
        "timestamp": datetime.utcnow()
    })

    return {
        "response": response.content
    }
