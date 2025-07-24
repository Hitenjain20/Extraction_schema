# Structured Extraction API

A production-ready FastAPI service that converts unstructured text into structured JSON following complex schemas with minimal constraints. This system intelligently handles documents ranging from simple forms to complex 150+ page reports with sophisticated nested schemas.

## 🎯 Key Features

- **Complex Schema Support**: Handles 3-7 nesting levels, 50-150 nested objects, 1000+ literals/enums
- **Large Document Processing**: Supports 50+ page documents up to 10MB with intelligent splitting
- **Adaptive Processing**: Automatically adjusts computational effort based on schema and document complexity
- **Confidence Scoring**: Identifies fields requiring human review with detailed confidence metrics
- **Flexible Input Methods**: URL-based processing, direct file uploads, and large document splitting
- **Async Processing**: Background job processing for long-running extractions

## 📁 Project Structure

```
extraction-api/
├── main.py                          # FastAPI application with all endpoints
├── utils/
│   └── extraction_pipeline.py       # Core extraction system logic
├── .env                             # Environment variables (create this)
├── pyproject.toml                   # UV package configuration
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [UV package manager](https://github.com/astral-sh/uv)
- Reducto API key

### Installation

```bash
# Clone the repository
git clone https://github.com/Hitenjain20/Extraction_schema.git
cd extraction-api

# Install dependencies with UV
uv add fastapi uvicorn pydantic python-multipart python-dotenv aiofiles reducto


```

### Configuration

Create a `.env` file in the root directory:

```bash
# Required: Your Reducto API key
REDUCTO_API_KEY=your_reducto_api_key_here

# Optional: API configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=info
```

### Running the API

```bash
# Start the development server
uv run python main.py

# Or with uvicorn directly
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **Server**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## 📚 API Endpoints

### Core Extraction Endpoints

#### `POST /extract-sync`
**Synchronous extraction for quick processing**

Best for: Small documents, simple schemas, immediate results needed

```bash
curl -X POST "http://localhost:8000/extract-sync" \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "https://example.com/contract.pdf",
    "json_schema": {
      "type": "object",
      "properties": {
        "contract_amount": {"type": "number"},
        "parties": {"type": "array", "items": {"type": "string"}},
        "effective_date": {"type": "string"}
      }
    },
    "system_prompt": "Extract contract details focusing on financial terms."
  }'
```

#### `POST /extract`
**Asynchronous extraction with job tracking**

Best for: Large documents, complex schemas, background processing

```bash
# Submit extraction job
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "https://example.com/large-report.pdf",
    "json_schema": { /* complex schema */ },
    "system_prompt": "Extract comprehensive data from this report."
  }'

# Returns: {"job_id": "uuid", "status": "queued", ...}
```

#### `POST /upload-and-extract`
**Direct file upload and processing**

Best for: Local files, when you don't have a publicly accessible URL

```bash
curl -X POST "http://localhost:8000/upload-and-extract" \
  -F "file=@/path/to/document.pdf" \
  -F 'schema={"type":"object","properties":{"title":{"type":"string"}}}' \
  -F 'system_prompt=Extract the document title and key information.'
```

#### `POST /extract-large-document`
**Intelligent document splitting for massive files**

Best for: 50+ page documents, multi-section reports, when you need section-wise extraction

```bash
curl -X POST "http://localhost:8000/extract-large-document" \
  -H "Content-Type: application/json" \
  -d '{
    "document_url": "https://example.com/150-page-report.pdf",
    "json_schema": { /* your schema */ },
    "split_descriptions": [
      {"name": "executive_summary", "description": "Executive summary section"},
      {"name": "financial_data", "description": "Financial performance data"},
      {"name": "recommendations", "description": "Strategic recommendations"}
    ],
    "system_prompt": "Extract key information from each section."
  }'
```

### Job Management Endpoints

#### `GET /jobs/{job_id}`
**Check extraction job status and retrieve results**

```bash
curl "http://localhost:8000/jobs/your-job-id"
```

Returns detailed job information including:
- Processing status (queued/processing/completed/failed)
- Extracted data when completed
- Confidence scores for each field
- Processing time and strategy used
- Fields flagged for human review

#### `GET /jobs`
**List all jobs with optional filtering**

```bash
# List recent jobs
curl "http://localhost:8000/jobs?limit=10"

# Filter by status
curl "http://localhost:8000/jobs?status=completed&limit=5"
```

#### `DELETE /jobs/{job_id}`
**Clean up completed jobs**

```bash
curl -X DELETE "http://localhost:8000/jobs/your-job-id"
```

### Analysis and Utility Endpoints

#### `POST /analyze-complexity`
**Analyze schema and document complexity without extraction**

Best for: Cost estimation, strategy planning, understanding processing requirements

```bash
curl -X POST "http://localhost:8000/analyze-complexity" \
  -H "Content-Type: application/json" \
  -d '{
    "json_schema": { /* your complex schema */ },
    "document_url": "https://example.com/document.pdf"
  }'
```

Returns:
- Schema complexity level (low/medium/high)
- Document complexity assessment
- Recommended processing strategy
- Estimated cost multiplier

#### `GET /sample-schema`
**Get a sample complex schema for testing**

```bash
curl "http://localhost:8000/sample-schema"
```

Returns a comprehensive schema example with multiple nesting levels, arrays, and enums.

#### `GET /health`
**API health check**

```bash
curl "http://localhost:8000/health"
```

## 🎛️ Processing Intelligence

### Automatic Strategy Selection

The system automatically analyzes your schema and document to choose the optimal processing approach:

- **Low Complexity**: Fast models, minimal overhead, synchronous processing
- **Medium Complexity**: Balanced approach with citations and validation
- **High Complexity**: Most accurate models, async processing, comprehensive citations

### Complexity Factors

**Schema Complexity:**
- Nesting levels (3-7+ levels)
- Total field count (50-150+ fields)
- Enum/literal count (100-1000+ options)

**Document Complexity:**
- Page count (20+ pages = medium, 50+ = high)
- Content density and structure
- Multi-section documents

## 📊 Response Format

All extraction endpoints return a consistent response structure:

```json
{
  "job_id": "unique-identifier",
  "status": "completed|processing|failed",
  "data": { /* extracted structured data */ },
  "confidence_scores": { /* field-level confidence ratings */ },
  "processing_time": 3.45,
  "strategy_used": {
    "ocr_mode": "standard|agentic",
    "model_preference": "fast|balanced|accurate",
    "estimated_cost_multiplier": 1.5
  },
  "citations": [ /* source citations for verification */ ],
  "low_confidence_fields": [ /* fields requiring human review */ ],
  "created_at": "2025-07-25T10:30:00Z",
  "completed_at": "2025-07-25T10:30:03Z"
}
```

## 🔄 Usage Patterns

### Pattern 1: Quick Document Processing
```python
import requests

response = requests.post("http://localhost:8000/extract-sync", json={
    "document_url": "https://example.com/invoice.pdf",
    "json_schema": {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total_amount": {"type": "number"},
            "due_date": {"type": "string"}
        }
    }
})

result = response.json()
print(f"Invoice Amount: ${result['data']['total_amount']}")
```

### Pattern 2: Large Document with Monitoring
```python
import requests
import time

# Submit job
response = requests.post("http://localhost:8000/extract", json={
    "document_url": "https://example.com/annual-report.pdf",
    "json_schema": complex_annual_report_schema
})

job_id = response.json()["job_id"]

# Monitor progress
while True:
    status_response = requests.get(f"http://localhost:8000/jobs/{job_id}")
    job_data = status_response.json()
    
    if job_data["status"] == "completed":
        print("Extraction completed!")
        print(f"Low confidence fields: {job_data['low_confidence_fields']}")
        break
    elif job_data["status"] == "failed":
        print(f"Extraction failed: {job_data['error_message']}")
        break
    
    time.sleep(5)  # Check every 5 seconds
```

### Pattern 3: Batch Processing with Complexity Analysis
```python
import requests

documents = [
    ("simple-form.pdf", simple_schema),
    ("complex-contract.pdf", complex_schema),
    ("massive-report.pdf", comprehensive_schema)
]

for doc_url, schema in documents:
    # Analyze complexity first
    complexity = requests.post("http://localhost:8000/analyze-complexity", json={
        "json_schema": schema,
        "document_url": doc_url
    }).json()
    
    # Choose endpoint based on complexity
    if complexity["schema_complexity"] == "high":
        # Use async for complex documents
        response = requests.post("http://localhost:8000/extract", json={
            "document_url": doc_url,
            "json_schema": schema
        })
        print(f"Submitted async job: {response.json()['job_id']}")
    else:
        # Use sync for simpler documents
        response = requests.post("http://localhost:8000/extract-sync", json={
            "document_url": doc_url,
            "json_schema": schema
        })
        print(f"Completed sync extraction: {response.json()['status']}")
```

## 🔒 Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `REDUCTO_API_KEY` | ✅ | Your Reducto API key for document processing | - |
| `API_HOST` | ❌ | Host to bind the API server | `0.0.0.0` |
| `API_PORT` | ❌ | Port for the API server | `8000` |
| `LOG_LEVEL` | ❌ | Logging level (debug/info/warning/error) | `info` |

## 🧪 Testing

Visit the interactive documentation at http://localhost:8000/docs to:
- Test all endpoints with sample data
- Explore request/response schemas
- Try different complexity scenarios
- Monitor job processing in real-time

## 🚀 Performance Characteristics

- **Simple schemas (< 20 fields)**: ~1-3 seconds
- **Medium schemas (20-100 fields)**: ~3-10 seconds  
- **Complex schemas (100+ fields)**: ~10-60 seconds
- **Large documents (50+ pages)**: ~30 seconds - 5 minutes
- **Massive documents with splitting**: ~2-15 minutes

Processing times vary based on document content, schema complexity, and chosen processing strategy.

