import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from groq import Groq
from pydantic import BaseModel, Field
from pypdf import PdfReader


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured."
    )


# ============================================================
# GROQ CLIENT
# ============================================================

client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# MODELS
# ============================================================

CHAT_MODEL = "openai/gpt-oss-120b"

RESUME_PARSER_MODEL = "llama-3.1-8b-instant"


# ============================================================
# PATH DEFINITIONS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BACKEND_DIR.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"

RESUME_PATH = BACKEND_DIR / "my_resume.pdf"


# ============================================================
# GLOBAL RESUME CACHE
# ============================================================

candidate_resume: dict = {}


# ============================================================
# PYDANTIC MODELS
# ============================================================

class Experience(BaseModel):

    company: str | None = None

    role: str | None = None

    duration: str | None = None

    description: str | None = None

    skills_used: list[str] = Field(
        default_factory=list
    )


class Resume(BaseModel):

    name: str | None = None

    email: str | None = None

    phone: str | None = None

    total_experience_years: float | None = None

    skills: list[str] = Field(
        default_factory=list
    )

    experiences: list[Experience] = Field(
        default_factory=list
    )

    education: list[str] = Field(
        default_factory=list
    )

    projects: list[str] = Field(
        default_factory=list
    )

    certifications: list[str] = Field(
        default_factory=list
    )


resume_schema = Resume.model_json_schema()


class ChatRequest(BaseModel):

    question: str


# ============================================================
# PDF READER
# ============================================================

def read_pdf(file_path: Path) -> str:

    """
    Extract text from the candidate's PDF resume.
    """

    if not file_path.exists():

        raise FileNotFoundError(
            f"Resume PDF not found at: {file_path}"
        )

    print(
        f"📄 Reading resume from: {file_path}"
    )

    start_time = time.perf_counter()

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"📄 PDF extraction completed in "
        f"{elapsed:.2f}s"
    )

    return text


# ============================================================
# LLM RESUME PARSER
# ============================================================

def parse_resume(resume_text: str) -> Resume:

    """
    Uses an LLM to convert the unstructured resume
    into structured Resume data.
    """

    start_time = time.perf_counter()

    print(
        f"🤖 Parsing resume using "
        f"{RESUME_PARSER_MODEL}..."
    )

    system_prompt = f"""
You are an expert resume parser.

Extract information from the candidate's resume.

Return ONLY valid JSON matching this schema:

{resume_schema}

IMPORTANT RULES:

1. Do not invent information.

2. Only extract information explicitly present
   in the resume.

3. If information is unavailable, return null.

4. If a list has no information, return an empty list.

5. Preserve the meaning of the original resume.

6. Do not add explanations outside the JSON.
"""

    response = client.chat.completions.create(

        model=RESUME_PARSER_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    "Parse the following resume:\n\n"
                    f"{resume_text}"
                )
            }
        ],

        response_format={
            "type": "json_object"
        },

        temperature=0
    )

    raw_output = (
        response
        .choices[0]
        .message
        .content
    )

    try:

        parsed_data = json.loads(
            raw_output
        )

    except json.JSONDecodeError as e:

        print(
            f"❌ Invalid JSON returned by LLM: {e}"
        )

        raise RuntimeError(
            "Resume parser returned invalid JSON."
        )

    parsed_resume = Resume(
        **parsed_data
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"✅ Resume parsing completed in "
        f"{elapsed:.2f}s"
    )

    return parsed_resume


# ============================================================
# INITIALIZE RESUME
# ============================================================

def initialize_resume():

    """
    Reads and parses the resume once during startup.

    The parsed resume is cached in memory.
    """

    global candidate_resume

    start_time = time.perf_counter()

    try:

        # ----------------------------------------------------
        # STEP 1: READ PDF
        # ----------------------------------------------------

        resume_text = read_pdf(
            RESUME_PATH
        )

        if not resume_text.strip():

            raise RuntimeError(
                "No text could be extracted from resume PDF."
            )


        # ----------------------------------------------------
        # STEP 2: PARSE RESUME
        # ----------------------------------------------------

        parsed_resume = parse_resume(
            resume_text
        )


        # ----------------------------------------------------
        # STEP 3: CACHE
        # ----------------------------------------------------

        candidate_resume["data"] = (
            parsed_resume
        )


        total_time = (
            time.perf_counter()
            - start_time
        )

        print(
            "\n======================================"
        )

        print(
            "✅ RESUME INITIALIZATION COMPLETE"
        )

        print(
            f"👤 Candidate: "
            f"{parsed_resume.name}"
        )

        print(
            f"⚡ Initialization time: "
            f"{total_time:.2f}s"
        )

        print(
            "======================================\n"
        )


    except Exception as e:

        print(
            f"❌ Error initializing resume: {e}"
        )


# ============================================================
# FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print(
        "\n🚀 Starting Hire Me AI..."
    )

    initialize_resume()

    yield

    print(
        "\n🛑 Hire Me AI shutting down..."
    )


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(

    title="Hire Me - AI Candidate Portfolio",

    description=(
        "AI-powered candidate portfolio using "
        "PDF processing, LLM-based resume extraction, "
        "structured data and streaming AI-powered HR Q&A."
    ),

    version="2.0.0",

    lifespan=lifespan
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def serve_frontend():

    index_file = (
        FRONTEND_DIR / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(

            status_code=404,

            detail=(
                "index.html not found in "
                f"{FRONTEND_DIR}"
            )
        )

    return FileResponse(
        index_file,
        media_type="text/html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    resume: Resume | None = (
        candidate_resume.get("data")
    )

    return {

        "status": "healthy",

        "resume_loaded":
            resume is not None,

        "candidate":
            resume.name if resume else None,

        "chat_model":
            CHAT_MODEL,

        "resume_parser_model":
            RESUME_PARSER_MODEL,

        "streaming":
            True,

        "resume_preview_endpoint":
            "/resume-preview",
        "resume_download_endpoint":
            "/download-resume"
    }


# ============================================================
# RESUME HELPERS
# ============================================================

def validate_resume_file() -> Path:
    """Validate that the resume exists and is a non-empty file."""
    if not RESUME_PATH.exists():
        print(f"❌ Resume not found: {RESUME_PATH}")
        raise HTTPException(
            status_code=404,
            detail=(
                "Resume PDF not found on server. "
                "Make sure my_resume.pdf is deployed inside the backend folder."
            )
        )

    if not RESUME_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="Resume path exists but is not a file."
        )

    file_size = RESUME_PATH.stat().st_size

    if file_size == 0:
        print("❌ Resume PDF exists but is empty.")
        raise HTTPException(
            status_code=500,
            detail="Resume PDF is empty."
        )

    print(f"📄 Resume ready: {RESUME_PATH} ({file_size} bytes)")
    return RESUME_PATH


# ============================================================
# RESUME PREVIEW
# ============================================================

@app.get("/resume-preview")
def preview_resume():
    """Open the resume in the browser PDF viewer."""
    resume_file = validate_resume_file()

    return FileResponse(
        path=resume_file,
        media_type="application/pdf",
        filename="Ankit_Kumar_Resume.pdf",
        headers={
            "Content-Disposition":
                'inline; filename="Ankit_Kumar_Resume.pdf"',
            "Cache-Control":
                "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Content-Type-Options": "nosniff"
        }
    )


# ============================================================
# RESUME DOWNLOAD
# ============================================================

@app.get("/download-resume")
def download_resume():
    """Force a direct PDF download."""
    resume_file = validate_resume_file()

    return FileResponse(
        path=resume_file,
        media_type="application/pdf",
        filename="Ankit_Kumar_Resume.pdf",
        headers={
            "Content-Disposition":
                'attachment; filename="Ankit_Kumar_Resume.pdf"',
            "Cache-Control":
                "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Content-Type-Options": "nosniff"
        }
    )


# ============================================================
# CANDIDATE INFO
# ============================================================

@app.get("/candidate-info")
def get_info():

    resume: Resume | None = (
        candidate_resume.get("data")
    )

    if not resume:

        return {

            "name":
                "Candidate",

            "skills":
                [],

            "total_experience_years":
                None
        }

    return {

        "name":
            resume.name,

        "skills":
            resume.skills,

        "total_experience_years":
            resume.total_experience_years
    }


# ============================================================
# BUILD SYSTEM PROMPT
# ============================================================

def build_system_prompt(
    resume: Resume
) -> str:

    resume_json = (
        resume.model_dump_json(
            indent=2
        )
    )

    return f"""
You are "Hire Me", an AI assistant representing
the job candidate:

{resume.name or "the applicant"}

You are speaking directly to an HR recruiter
or hiring manager.

CANDIDATE RESUME DETAILS:

{resume_json}


============================================================
ACCURACY RULES
============================================================

1. Answer strictly using the candidate information above.

2. Never invent or hallucinate information.

3. Do not make assumptions.

4. If information is not available in the resume,
say:

"I don't have that specific detail in my resume,
but I'd be happy to discuss it further during
an interview!"


============================================================
RESPONSE STYLE
============================================================

Your response is displayed directly to an HR recruiter.

Make every response:

• Professional
• Clear
• Concise
• Easy to scan
• Recruiter-friendly


============================================================
FORMATTING RULES
============================================================

DO NOT use Markdown.

DO NOT use:

**
***
#
##
###
- **
* **
__text__

Do not put asterisks around words.

Do not create Markdown headings.

Use plain-text headings.

Use this bullet character:

•

Example:

Professional Summary

• Computer Science Engineering graduate
• Strong background in Python and SQL
• Experience with Power BI and analytics


Skills

• Python
• SQL
• Power BI
• Excel


============================================================
HEADINGS
============================================================

When appropriate, use:

Professional Summary

Technical Skills

Education

Experience

Projects

Certifications

Contact Information


============================================================
PROJECTS
============================================================

When discussing projects:

Project Name

• What the project does
• Technologies used
• Important functionality
• Candidate's contribution


============================================================
SKILLS
============================================================

Organize skills into categories where useful.

Example:

Programming

• Python
• C++
• JavaScript

Data & Analytics

• SQL
• Power BI
• Excel

Development

• React
• Node.js
• FastAPI


============================================================
ABOUT THE CANDIDATE
============================================================

For questions such as:

"Tell me about the candidate"

"Tell me about Ankit"

"Walk me through the resume"

provide a professional summary with:

Professional Summary

• Education
• Technical background
• Key skills
• Projects
• Career focus


============================================================
RESPONSE LENGTH
============================================================

Simple questions:

3–6 bullet points.

Detailed questions:

Use headings and bullet points.

Avoid unnecessary repetition.

Keep the answer useful for an HR recruiter.
"""


# ============================================================
# STREAMING CHAT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    """
    Streams the LLM response token-by-token.

    The frontend progressively displays the response.
    """

    request_start = time.perf_counter()


    # --------------------------------------------------------
    # GET CACHED RESUME
    # --------------------------------------------------------

    resume: Resume | None = (
        candidate_resume.get("data")
    )

    if not resume:

        raise HTTPException(

            status_code=500,

            detail=(
                "Resume data is not loaded. "
                "Ensure my_resume.pdf exists "
                "and resume initialization succeeded."
            )
        )


    # --------------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------------

    question = (
        request.question.strip()
    )

    if not question:

        raise HTTPException(

            status_code=400,

            detail="Question cannot be empty."
        )


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    system_prompt = build_system_prompt(
        resume
    )


    # --------------------------------------------------------
    # STREAM GENERATOR
    # --------------------------------------------------------

    def generate():

        llm_start = time.perf_counter()

        first_token_time = None

        total_characters = 0


        try:

            print(
                f"\n💬 Streaming question: "
                f"{question}"
            )


            # ------------------------------------------------
            # GROQ STREAM
            # ------------------------------------------------

            stream = (
                client
                .chat
                .completions
                .create(

                    model=CHAT_MODEL,

                    messages=[
                        {
                            "role":
                                "system",

                            "content":
                                system_prompt
                        },
                        {
                            "role":
                                "user",

                            "content":
                                question
                        }
                    ],

                    temperature=0.2,

                    max_tokens=500,

                    stream=True
                )
            )


            # ------------------------------------------------
            # READ STREAM
            # ------------------------------------------------

            for chunk in stream:

                if not chunk.choices:

                    continue


                delta = (
                    chunk
                    .choices[0]
                    .delta
                )


                content = getattr(
                    delta,
                    "content",
                    None
                )


                if not content:

                    continue


                # ------------------------------------------------
                # FIRST TOKEN
                # ------------------------------------------------

                if first_token_time is None:

                    first_token_time = (
                        time.perf_counter()
                        - llm_start
                    )

                    print(
                        f"⚡ First token: "
                        f"{first_token_time:.2f}s"
                    )


                # ------------------------------------------------
                # SEND RAW TEXT TO FRONTEND FORMATTER
                # ------------------------------------------------
                # Do not strip Markdown per streaming chunk.
                # A marker such as ** can be split across chunks.
                # The frontend formats the complete accumulated text.


                total_characters += (
                    len(content)
                )


                # ------------------------------------------------
                # SEND CONTENT
                # ------------------------------------------------

                yield content


            # ------------------------------------------------
            # FINAL TIMING
            # ------------------------------------------------

            total_time = (
                time.perf_counter()
                - request_start
            )


            print(
                "✅ Streaming completed"
            )

            print(
                f"⏱️ Total time: "
                f"{total_time:.2f}s"
            )

            print(
                f"📝 Characters: "
                f"{total_characters}"
            )

            print(
                "======================================\n"
            )


        except Exception as e:

            print(
                f"❌ Streaming error: {e}"
            )


            yield (
                "\n\nSorry, I encountered an error "
                "while generating the response."
            )


    # --------------------------------------------------------
    # RETURN STREAM
    # --------------------------------------------------------

    return StreamingResponse(

        generate(),

        media_type="text/plain; charset=utf-8",

        headers={

            "Cache-Control":
                "no-cache, no-transform",

            "Connection":
                "keep-alive",

            "X-Accel-Buffering":
                "no",

            "Access-Control-Allow-Origin":
                "*"
        }
    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),
        reload=True
    )
