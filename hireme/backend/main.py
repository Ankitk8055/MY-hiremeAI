from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader

# Load environment variables (.env file inside backend/)
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "openai/gpt-oss-120b"

# Path definitions
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESUME_PATH = BACKEND_DIR / "my_resume.pdf"

# Global store to cache parsed resume in memory
candidate_resume: dict = {}


# --- Pydantic Data Models ---
class Experience(BaseModel):
    company: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

resume_schema = Resume.model_json_schema()

class ChatRequest(BaseModel):
    question: str


# --- Helper Functions ---
def read_pdf(file_path: Path) -> str:
    """Reads and extracts text from a PDF file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Resume PDF not found at path: {file_path}")
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def parse_resume(resume_text: str) -> Resume:
    """Parses raw text into structured JSON using Groq LLM."""
    system_prompt = f"""
    You are an expert resume parser. Extract information based on context.
    Return ONLY valid JSON matching this schema:
    {resume_schema}
    
    Important rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse the following resume:\n\n{resume_text}"}
        ],
        response_format={"type": "json_object"}
    )
    raw_output = response.choices[0].message.content
    return Resume(**json.loads(raw_output))


# --- FastAPI Lifespan (Startup Task) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes on application startup to load and parse my_resume.pdf once."""
    global candidate_resume
    try:
        print(f"📄 Reading resume from: {RESUME_PATH}")
        text = read_pdf(RESUME_PATH)
        print("🤖 Parsing resume with Groq AI...")
        parsed = parse_resume(text)
        candidate_resume["data"] = parsed
        print(f"✅ Resume successfully loaded for: {parsed.name}")
    except Exception as e:
        print(f"❌ Error initializing resume parsing: {e}")
    yield


# --- App Setup ---
app = FastAPI(title="AI Candidate Portfolio", lifespan=lifespan)

# Enable CORS for local development and production frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Routes ---
@app.get("/")
def serve_frontend():
    """Serves the index.html from the frontend/ directory."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"index.html not found in frontend directory: {FRONTEND_DIR}"
        )
    return FileResponse(index_file)

@app.get("/candidate-info")
def get_info():
    """Returns candidate name and skills for header customization."""
    resume: Resume | None = candidate_resume.get("data")
    if not resume:
        return {"name": "Candidate", "skills": []}
    return {
        "name": resume.name,
        "skills": resume.skills,
        "total_experience_years": resume.total_experience_years
    }

@app.post("/chat")
def chat(request: ChatRequest):
    """Answers HR questions based strictly on candidate resume data."""
    resume: Resume | None = candidate_resume.get("data")
    if not resume:
        raise HTTPException(
            status_code=500, 
            detail="Resume data is not loaded. Ensure my_resume.pdf exists in backend/."
        )

    system_prompt = f"""
You are an AI assistant representing the job candidate ({resume.name or "the applicant"}).
You are speaking directly to an HR recruiter or hiring manager during an interview process.

Candidate Resume Details:
{resume.model_dump_json(indent=2)}

Rules:
1. Answer strictly and accurately using only the information above.
2. Be professional, direct, and engaging.
3. Do not invent or hallucinate facts not present in the resume.
4. If a question asks about something not mentioned in the resume, reply:
   "I don't have that specific detail in my resume, but I'd be happy to discuss it further during an interview!"
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API Error: {str(e)}")