
import logging
import pickle
import re
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PyPDF2 import PdfReader


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Screening and Job Recommendation",
    description=(
        "An ML-powered application for resume categorization, "
        "job recommendations, and resume information extraction."
    ),
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEMPLATE_DIR = BASE_DIR / "templates"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

def load_pickle(filename):
    """Load a trained model or vectorizer from the models directory."""
    model_path = MODEL_DIR / filename

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required model file is missing: {model_path}"
        )

    with model_path.open("rb") as file:
        return pickle.load(file)


try:
    rf_classifier_categorization = load_pickle(
        "rf_classifier_categorization.pkl"
    )

    tfidf_vectorizer_categorization = load_pickle(
        "tfidf_vectorizer_categorization.pkl"
    )

    rf_classifier_job_recommendation = load_pickle(
        "rf_classifier_job_recommendation.pkl"
    )

    tfidf_vectorizer_job_recommendation = load_pickle(
        "tfidf_vectorizer_job_recommendation.pkl"
    )

    MODELS_LOADED = True
    logger.info("All four ML model files loaded successfully.")

except Exception:
    MODELS_LOADED = False
    logger.exception(
        "Could not load the ML models. Check the models directory "
        "and the scikit-learn version used to create the pickle files."
    )


# ============================================================
# RESUME TEXT CLEANING
# ============================================================

def cleanResume(txt):
    """Clean resume text before passing it to the TF-IDF vectorizer."""
    clean_text = re.sub(r"https?://\S+|www\.\S+", " ", txt)
    clean_text = re.sub(r"\bRT\b|cc", " ", clean_text)
    clean_text = re.sub(r"#\S+|@\S+", " ", clean_text)

    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    clean_text = re.sub(
        "[" + re.escape(punctuation) + "]",
        " ",
        clean_text,
    )

    # Preserve the original cleaning approach.
    clean_text = re.sub(r"[^\x00-\x7f]", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text)

    return clean_text.strip()


# ============================================================
# CHECK MODEL AVAILABILITY
# ============================================================

def ensure_models_loaded():
    """Return an error if the required models failed to load."""
    if not MODELS_LOADED:
        raise RuntimeError(
            "The trained models are unavailable. Check the models "
            "directory and model compatibility."
        )


# ============================================================
# RESUME CATEGORY PREDICTION
# ============================================================

def predict_category(resume_text):
    ensure_models_loaded()

    cleaned_text = cleanResume(resume_text)

    if not cleaned_text:
        raise ValueError("The resume does not contain usable text.")

    resume_tfidf = tfidf_vectorizer_categorization.transform(
        [cleaned_text]
    )

    predicted_category = (
        rf_classifier_categorization.predict(resume_tfidf)[0]
    )

    return str(predicted_category)


# ============================================================
# JOB RECOMMENDATION
# ============================================================

def job_recommendation(resume_text):
    ensure_models_loaded()

    cleaned_text = cleanResume(resume_text)

    if not cleaned_text:
        raise ValueError("The resume does not contain usable text.")

    resume_tfidf = tfidf_vectorizer_job_recommendation.transform(
        [cleaned_text]
    )

    recommended_job = (
        rf_classifier_job_recommendation.predict(resume_tfidf)[0]
    )

    return str(recommended_job)


# ============================================================
# PDF TO TEXT
# ============================================================

def pdf_to_text(file):
    """Extract selectable text from a PDF file."""
    reader = PdfReader(file)

    if reader.is_encrypted:
        raise ValueError(
            "Password-protected PDF files are not supported."
        )

    pages_text = []

    for page in reader.pages:
        pages_text.append(page.extract_text() or "")

    return "\n".join(pages_text)


# ============================================================
# EXTRACT PHONE NUMBER
# ============================================================

def extract_contact_number_from_resume(text):
    """Extract a likely phone number from resume text."""
    pattern = (
        r"(?<!\w)"
        r"(?:\+?\d{1,3}[-.\s]?)?"
        r"(?:\(?\d{3,5}\)?[-.\s]?)?"
        r"\d{3,5}[-.\s]?\d{3,5}"
        r"(?!\w)"
    )

    match = re.search(pattern, text)

    return match.group().strip() if match else None


# ============================================================
# EXTRACT EMAIL
# ============================================================

def extract_email_from_resume(text):
    """Extract the first email address found in the resume."""
    pattern = (
        r"\b[A-Za-z0-9._%+-]+"
        r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    )

    match = re.search(pattern, text)

    return match.group(0) if match else None


# ============================================================
# EXTRACT SKILLS
# ============================================================

SKILLS_LIST = [
    "Python", "SQL", "Java", "C++", "JavaScript",
    "HTML", "CSS", "React", "Angular", "Node.js",
    "MongoDB", "Express.js", "Git", "Linux",

    "Data Analysis", "Data Science", "Data Analytics",
    "Machine Learning", "Deep Learning",
    "Natural Language Processing", "Computer Vision",
    "Statistics", "Data Visualization", "Predictive Analytics",
    "Data Mining", "Text Mining", "Recommendation Systems",
    "Neural Networks", "Reinforcement Learning", "Generative AI",

    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Plotly",
    "Scikit-learn", "TensorFlow", "Keras", "PyTorch",
    "NLTK", "OpenCV", "XGBoost", "Random Forest",
    "Decision Trees", "Support Vector Machines",
    "Linear Regression", "Logistic Regression",
    "K-Means Clustering",

    "Flask", "FastAPI", "Django", "RESTful APIs",
    "GraphQL", "Microservices", "Streamlit",
    "MySQL", "PostgreSQL", "SQLite", "Redis",
    "Apache Spark", "Apache Hadoop", "Apache Kafka",
    "ETL", "Data Engineering", "Data Warehousing",

    "AWS", "Amazon Web Services", "Microsoft Azure",
    "Google Cloud Platform", "Docker", "Kubernetes",
    "CI/CD", "DevOps", "MLOps", "GitHub Actions",

    "Tableau", "Power BI", "Excel", "Business Intelligence",
    "Project Management", "Communication", "Research",
    "Agile Methodology", "Scrum", "Software Development",
    "Web Development", "Cybersecurity", "Selenium",
    "API Testing", "Quality Assurance",

    "C#", "PHP", "Ruby", "Swift", "Kotlin", "R",
    "Spring", "Hibernate", "Flutter", "React Native",
    "Firebase", "Elasticsearch", "Cassandra",
    "Blockchain", "NLP", "OCR", "Image Processing",
    "Speech Recognition", "Fraud Detection",
    "Time Series Analysis", "A/B Testing",
]


def extract_skills_from_resume(text):
    """Find skills from the predefined skills dictionary."""
    skills = []

    for skill in SKILLS_LIST:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

        if re.search(pattern, text, re.IGNORECASE):
            skills.append(skill)

    return skills


# ============================================================
# EXTRACT EDUCATION
# ============================================================

EDUCATION_KEYWORDS = [
    "Computer Science",
    "Computer Engineering",
    "Information Technology",
    "Software Engineering",
    "Data Science",
    "Data Analytics",
    "Artificial Intelligence",
    "Machine Learning",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Electronics Engineering",
    "Electronics and Communication",
    "Biomedical Engineering",
    "Biotechnology",
    "Mathematics",
    "Statistics",
    "Physics",
    "Chemistry",
    "Economics",
    "Finance",
    "Accounting",
    "Business Administration",
    "Human Resource Management",
    "Marketing",
    "Psychology",
    "Commerce",
    "Bachelor of Technology",
    "Bachelor of Engineering",
    "Bachelor of Science",
    "Bachelor of Computer Applications",
    "Master of Technology",
    "Master of Engineering",
    "Master of Science",
    "Master of Computer Applications",
    "Master of Business Administration",
    "Bachelor of Commerce",
    "Master of Commerce",
    "Bachelor of Arts",
    "Master of Arts",
    "Diploma",
]


def extract_education_from_resume(text):
    """Find education-related keywords in the resume."""
    education = []

    for keyword in EDUCATION_KEYWORDS:
        pattern = re.compile(
            r"(?<!\w)" + re.escape(keyword) + r"(?!\w)",
            re.IGNORECASE,
        )

        match = pattern.search(text)

        if match:
            education.append(match.group(0).strip())

    # Remove duplicate entries while preserving order.
    return list(dict.fromkeys(education))


# ============================================================
# EXTRACT NAME
# ============================================================

def extract_name_from_resume(text):
    """
    Heuristic name extraction.
    This may not identify every name correctly.
    """
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    name_pattern = re.compile(
        r"^[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){1,3}$"
    )

    for line in lines[:10]:
        if name_pattern.fullmatch(line):
            return line

    return None


# ============================================================
# HTML TEMPLATE RESPONSE HELPERS
# ============================================================

def render_home(request, context=None, status_code=200):
    """Render the resume upload page and optional results."""
    return templates.TemplateResponse(
        request=request,
        name="resume.html",
        context=context or {},
        status_code=status_code,
    )


# ============================================================
# API ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main resume screening interface."""
    return render_home(request)


@app.get("/health")
async def health():
    """Report whether the application and models are available."""
    return JSONResponse(
        status_code=200 if MODELS_LOADED else 503,
        content={
            "status": "healthy" if MODELS_LOADED else "degraded",
            "models_loaded": MODELS_LOADED,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_resume(
    request: Request,
    resume: UploadFile = File(...),
):
    """Process a PDF/TXT resume and render prediction results."""

    try:
        # Validate the uploaded filename.
        if not resume.filename:
            return render_home(
                request,
                {"message": "Please select a resume file."},
                status_code=400,
            )

        filename = resume.filename.lower()

        if not filename.endswith((".pdf", ".txt")):
            return render_home(
                request,
                {
                    "message": (
                        "Invalid file format. "
                        "Please upload a PDF or TXT file."
                    )
                },
                status_code=400,
            )

        # Read the upload.
        contents = await resume.read()

        if not contents:
            return render_home(
                request,
                {"message": "The uploaded file is empty."},
                status_code=400,
            )

        # Extract text from PDF or TXT.
        try:
            if filename.endswith(".pdf"):
                resume_text = pdf_to_text(BytesIO(contents))
            else:
                resume_text = contents.decode("utf-8-sig")

        except ValueError as exc:
            return render_home(
                request,
                {"message": str(exc)},
                status_code=400,
            )

        except Exception:
            logger.exception("Could not read uploaded resume.")

            return render_home(
                request,
                {
                    "message": (
                        "Unable to read the file. "
                        "Please check that it is a valid PDF or TXT."
                    )
                },
                status_code=400,
            )

        if not resume_text.strip():
            return render_home(
                request,
                {
                    "message": (
                        "No readable text was found. "
                        "Scanned PDFs may require OCR."
                    )
                },
                status_code=400,
            )

        # Run ML predictions.
        try:
            predicted_category = predict_category(resume_text)
            recommended_job = job_recommendation(resume_text)

        except Exception:
            logger.exception("Model prediction failed.")

            return render_home(
                request,
                {
                    "message": (
                        "Prediction failed. Check that all four model "
                        "files exist and are compatible with the "
                        "installed scikit-learn version."
                    )
                },
                status_code=500,
            )

        # Extract resume information.
        name = extract_name_from_resume(resume_text)
        email = extract_email_from_resume(resume_text)
        phone = extract_contact_number_from_resume(resume_text)
        skills = extract_skills_from_resume(resume_text)
        education = extract_education_from_resume(resume_text)

        # Pass all values expected by the HTML template.
        context = {
            "filename": resume.filename,
            "predicted_category": predicted_category,
            "recommended_job": recommended_job,
            "name": name,
            "email": email,
            "phone": phone,
            "extracted_skills": skills,
            "extracted_education": education,
            "message": None,
        }

        return render_home(request, context)

    finally:
        await resume.close()


# ============================================================
# RUN DIRECTLY (OPTIONAL)
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
