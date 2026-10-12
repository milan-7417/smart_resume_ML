# Resume Screening and Job Recommendation

An end-to-end machine-learning application that analyzes resumes and provides an initial candidate profile. Upload a selectable-text PDF or TXT resume to receive:

- A predicted professional category
- A recommended job type
- Extracted contact details, including name, email, and phone number
- Detected skills and education-related keywords

The application combines TF-IDF text vectorization, Random Forest classifiers, PDF/text extraction, and a browser-based FastAPI interface.

> **Important:** This project is intended as an assistive screening tool and should not be used as the sole basis for employment decisions. Predictions and extracted fields should be reviewed by a qualified human.

## Features

- **Resume upload:** Accepts `.pdf` and `.txt` files.
- **Resume categorization:** Classifies resume text using a pre-trained TF-IDF and Random Forest pipeline.
- **Job recommendation:** Predicts a suitable job type using a separate trained pipeline.
- **Information extraction:** Uses transparent pattern- and keyword-based heuristics for:
  - Name
  - Email address
  - Phone number
  - Skills from the built-in skills dictionary
  - Education terms from the built-in education dictionary
- **Health monitoring:** Exposes `/health` to report application and model availability.
- **Interactive UI:** Renders results in a responsive Jinja2/HTML interface.

## How It Works

1. The user uploads a PDF or TXT resume through the web interface.
2. Text is extracted from the file. Password-protected PDFs and image-only PDFs are not supported.
3. Resume text is normalized by removing URLs, social tags, punctuation, non-ASCII characters, and excess whitespace.
4. The cleaned text is transformed with the appropriate TF-IDF vectorizer.
5. Random Forest classifiers predict the resume category and recommended job type.
6. Regular expressions and curated keyword lists extract additional resume information.
7. The predictions and extracted fields are rendered on the results page.

## Technology Stack

- **Backend:** Python, FastAPI, Uvicorn
- **Templating:** Jinja2
- **Machine learning:** scikit-learn, TF-IDF, Random Forest
- **Document processing:** PyPDF2
- **Data processing:** pandas, NumPy
- **Exploration and training:** Jupyter notebooks
- **Frontend:** HTML, CSS, and vanilla JavaScript

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── models/
│   ├── rf_classifier_categorization.pkl
│   ├── tfidf_vectorizer_categorization.pkl
│   ├── rf_classifier_job_recommendation.pkl
│   └── tfidf_vectorizer_job_recommendation.pkl
├── templates/
│   └── resume.html
├── data/
│   ├── clean_resume_data.csv
│   └── jobs_dataset_with_features (2).csv
├── Extracted Info and hiring process.ipynb
├── Resume Catogorization prediction.ipynb
└── Resume Job Recommendation System.ipynb
```

The serialized model files in `models/` are required at application startup. Keep them in that directory unless the model-loading logic in `app.py` is updated.

## Prerequisites

- Python 3.10 or newer
- A virtual environment
- The dependencies listed in `requirements.txt`
- The four pre-trained model files in `models/`

The web application also requires the FastAPI runtime packages used by `app.py`: `fastapi`, `uvicorn`, `jinja2`, and `python-multipart`.

## Installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/611noorsaeed/Resume-Screening-with-Machine-Learning-Job-Recommendations-Parsing-Categorization.git
cd Resume-Screening-with-Machine-Learning-Job-Recommendations-Parsing-Categorization

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn jinja2 python-multipart
```

## Running the Application

Start the development server with:

```bash
uvicorn app:app --reload
```

Alternatively, run the application module directly:

```bash
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in a browser, upload a resume, and review the results.

The interactive OpenAPI documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## API Endpoints

### `GET /`

Renders the resume upload and results interface.

### `GET /health`

Returns model availability:

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

The endpoint returns HTTP `503` when one or more required model files cannot be loaded.

### `POST /predict`

Accepts a multipart form upload with the field name `resume`.

Supported file types:

- `.pdf` — text is extracted with PyPDF2
- `.txt` — decoded as UTF-8

The endpoint returns the HTML interface populated with predictions and extracted information. Invalid, empty, encrypted, or unreadable files produce an explanatory error message.

## Model Artifacts

The application loads these files when it starts:

| File | Purpose |
| --- | --- |
| `tfidf_vectorizer_categorization.pkl` | Converts resume text into features for category prediction |
| `rf_classifier_categorization.pkl` | Predicts the professional category |
| `tfidf_vectorizer_job_recommendation.pkl` | Converts resume text into features for job recommendation |
| `rf_classifier_job_recommendation.pkl` | Predicts the recommended job type |

Because these are pickle files, load them only from a trusted source. Model files created with an incompatible scikit-learn version may fail to load or produce incompatible behavior.

## Research and Development Notebooks

The notebooks document exploratory and modeling work:

- `Resume Catogorization prediction.ipynb` — resume category prediction
- `Resume Job Recommendation System.ipynb` — job recommendation workflow
- `Extracted Info and hiring process.ipynb` — resume information extraction and hiring-process exploration

To work with the notebooks:

```bash
pip install jupyter
jupyter notebook
```

## Limitations

- Scanned or image-only PDFs require OCR and are not processed by the current implementation.
- Name, phone, skills, and education extraction are heuristic and may return missing or inaccurate values.
- The skills and education dictionaries are maintained in `app.py`; uncommon terms may not be detected.
- The model predictions depend on the training data and the compatibility of the installed scikit-learn version with the serialized artifacts.
- The application currently performs in-memory processing and does not provide authentication, persistence, audit logging, or a production file-upload policy.
- Do not upload confidential resumes to an environment that has not been secured for sensitive personal data.

## Production Considerations

Before deploying this project for real hiring workflows, consider:

- Pinning all runtime dependencies, including FastAPI-related packages
- Adding file-size limits, request timeouts, authentication, and rate limiting
- Adding OCR support only where appropriate and with explicit privacy controls
- Replacing pickle loading with a controlled model registry or another reviewed artifact format
- Adding automated tests for extraction, upload validation, health checks, and model compatibility
- Implementing access controls, encryption, retention policies, and audit logging for personal data
- Evaluating model quality, bias, fairness, and drift on representative data

## Contributing

Contributions are welcome. A typical workflow is:

1. Create a feature branch.
2. Make a focused change.
3. Update the documentation when behavior changes.
4. Run the application and verify the relevant workflow.
5. Open a pull request describing the change and validation performed.

## License

No license file is currently included in this repository. Add an explicit license before distributing or reusing the project.