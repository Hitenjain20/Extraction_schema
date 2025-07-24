from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import json
import uuid
import tempfile
import os
from pathlib import Path
import asyncio
import logging
from datetime import datetime

# Import your extraction system
# Note: Replace this with your actual module name
try:
    from utils.extraction_pipeline import StructuredExtractionSystem, ProcessingStrategy, ExtractionResult, ProcessingComplexity
except ImportError:
    print("Please create extraction_system.py with your StructuredExtractionSystem class")
    import sys
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Structured Extraction API",
    description="Convert unstructured text into structured JSON following complex schemas",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global extraction system instance
extraction_system = None

# In-memory job storage (use Redis/database in production)
job_storage = {}

# Pydantic models for API
class ExtractionRequest(BaseModel):
    document_url: str = Field(..., description="URL to the document to process")
    json_schema: Dict[str, Any] = Field(..., description="JSON schema defining desired output structure")
    system_prompt: str = Field(
        default="Be precise and thorough. Extract all relevant information from the document.",
        description="Instructions for the extraction model"
    )
    force_strategy: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Override automatic strategy selection"
    )

class LargeDocumentRequest(BaseModel):
    document_url: str = Field(..., description="URL to the large document")
    json_schema: Dict[str, Any] = Field(..., description="JSON schema for extraction")
    split_descriptions: List[Dict[str, str]] = Field(
        ...,
        description="List of section descriptions for document splitting"
    )
    system_prompt: str = Field(
        default="Be precise and thorough.",
        description="Instructions for extraction"
    )

class ExtractionResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: processing, completed, failed")
    data: Optional[Union[Dict[str, Any], List[Any]]] = Field(default=None, description="Extracted structured data")
    confidence_scores: Optional[Dict[str, Any]] = Field(default=None, description="Confidence scores for extracted fields")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    strategy_used: Optional[Dict[str, Any]] = Field(default=None, description="Processing strategy that was used")
    citations: Optional[List[Dict]] = Field(default=None, description="Source citations for extracted data")
    low_confidence_fields: Optional[List[str]] = Field(default=None, description="Fields flagged for human review")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    created_at: Optional[datetime] = Field(default=None, description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Job completion timestamp")

class ComplexityAnalysisRequest(BaseModel):
    json_schema: Dict[str, Any] = Field(..., description="JSON schema to analyze")
    document_url: Optional[str] = Field(default=None, description="Optional document URL for complexity analysis")

class ComplexityAnalysisResponse(BaseModel):
    schema_complexity: str = Field(..., description="Schema complexity level: low, medium, high")
    document_complexity: Optional[str] = Field(default=None, description="Document complexity level if URL provided")
    recommended_strategy: Dict[str, Any] = Field(..., description="Recommended processing strategy")
    estimated_cost_multiplier: float = Field(..., description="Estimated cost multiplier")
    analysis_details: Dict[str, Any] = Field(..., description="Detailed complexity analysis")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server timestamp")

# Dependency to get extraction system
def get_extraction_system() -> StructuredExtractionSystem:
    global extraction_system
    if extraction_system is None:
        extraction_system = StructuredExtractionSystem()
    return extraction_system

# Background task for processing
async def process_extraction_job(
    job_id: str,
    request: ExtractionRequest,
    system: StructuredExtractionSystem
):
    """Background task to process extraction jobs."""
    try:
        job_storage[job_id]["status"] = "processing"
        
        # Convert force_strategy dict back to ProcessingStrategy if provided
        force_strategy = None
        if request.force_strategy:
            force_strategy = ProcessingStrategy(**request.force_strategy)
        
        # Run extraction
        result = system.extract_structured_data(
            document_url=request.document_url,
            schema=request.json_schema,
            system_prompt=request.system_prompt,
            force_strategy=force_strategy
        )
        
        # Handle the case where data might be different types
        extracted_data = result.data
        if not isinstance(extracted_data, (dict, list, type(None))):
            # Convert to string if it's some other type
            extracted_data = str(extracted_data)
        
        # Update job with results
        job_storage[job_id].update({
            "status": "completed",
            "data": extracted_data,
            "confidence_scores": result.confidence_scores,
            "processing_time": result.processing_time,
            "strategy_used": {
                "ocr_mode": result.strategy_used.ocr_mode,
                "array_extract": result.strategy_used.array_extract,
                "model_preference": result.strategy_used.model_preference,
                "generate_citations": result.strategy_used.generate_citations,
                "use_async": result.strategy_used.use_async,
                "estimated_cost_multiplier": result.strategy_used.estimated_cost_multiplier
            },
            "citations": result.citations,
            "low_confidence_fields": result.low_confidence_fields,
            "completed_at": datetime.now()
        })
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job_storage[job_id].update({
            "status": "failed",
            "error_message": str(e),
            "completed_at": datetime.now()
        })

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/extract", response_model=ExtractionResponse)
async def extract_structured_data(
    request: ExtractionRequest,
    background_tasks: BackgroundTasks,
    system: StructuredExtractionSystem = Depends(get_extraction_system)
):
    """
    Extract structured data from a document URL following the provided schema.
    
    This endpoint processes documents asynchronously and returns a job ID.
    Use the /jobs/{job_id} endpoint to check status and retrieve results.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job in storage
    job_storage[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(),
        "request": request.dict()
    }
    
    # Add background task
    background_tasks.add_task(process_extraction_job, job_id, request, system)
    
    return ExtractionResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.now()
    )

@app.post("/extract-sync", response_model=ExtractionResponse)
async def extract_structured_data_sync(
    request: ExtractionRequest,
    system: StructuredExtractionSystem = Depends(get_extraction_system)
):
    """
    Extract structured data synchronously (for smaller documents/schemas).
    
    Use this endpoint for quick extractions where you can wait for the response.
    For large documents or complex schemas, use the async /extract endpoint.
    """
    job_id = str(uuid.uuid4())
    
    try:
        # Convert force_strategy dict back to ProcessingStrategy if provided
        force_strategy = None
        if request.force_strategy:
            force_strategy = ProcessingStrategy(**request.force_strategy)
        
        # Run extraction synchronously
        result = system.extract_structured_data(
            document_url=request.document_url,
            schema=request.json_schema,
            system_prompt=request.system_prompt,
            force_strategy=force_strategy
        )
        
        # Handle the case where data might be different types
        extracted_data = result.data
        if not isinstance(extracted_data, (dict, list, type(None))):
            # Convert to string if it's some other type
            extracted_data = str(extracted_data)
        
        return ExtractionResponse(
            job_id=job_id,
            status="completed",
            data=extracted_data,
            confidence_scores=result.confidence_scores,
            processing_time=result.processing_time,
            strategy_used={
                "ocr_mode": result.strategy_used.ocr_mode,
                "array_extract": result.strategy_used.array_extract,
                "model_preference": result.strategy_used.model_preference,
                "generate_citations": result.strategy_used.generate_citations,
                "use_async": result.strategy_used.use_async,
                "estimated_cost_multiplier": result.strategy_used.estimated_cost_multiplier
            },
            citations=result.citations,
            low_confidence_fields=result.low_confidence_fields,
            created_at=datetime.now(),
            completed_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Sync extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-and-extract")
async def upload_and_extract(
    file: UploadFile = File(...),
    schema: str = Form(..., description="JSON schema as string"),
    system_prompt: str = Form(
        default="Be precise and thorough.",
        description="Instructions for extraction"
    ),
    system: StructuredExtractionSystem = Depends(get_extraction_system)
):
    """
    Upload a file and extract structured data from it.
    
    Supports various file formats including PDF, DOCX, images, etc.
    """
    job_id = str(uuid.uuid4())
    
    try:
        # Parse schema from string
        schema_dict = json.loads(schema)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process the file
            result = system.upload_and_extract(
                file_path=tmp_file_path,
                schema=schema_dict,
                system_prompt=system_prompt
            )
            
            # Handle the case where data might be different types
            extracted_data = result.data
            if not isinstance(extracted_data, (dict, list, type(None))):
                # Convert to string if it's some other type
                extracted_data = str(extracted_data)
            
            return ExtractionResponse(
                job_id=job_id,
                status="completed",
                data=extracted_data,
                confidence_scores=result.confidence_scores,
                processing_time=result.processing_time,
                strategy_used={
                    "ocr_mode": result.strategy_used.ocr_mode,
                    "array_extract": result.strategy_used.array_extract,
                    "model_preference": result.strategy_used.model_preference,
                    "generate_citations": result.strategy_used.generate_citations,
                    "use_async": result.strategy_used.use_async,
                    "estimated_cost_multiplier": result.strategy_used.estimated_cost_multiplier
                },
                citations=result.citations,
                low_confidence_fields=result.low_confidence_fields,
                created_at=datetime.now(),
                completed_at=datetime.now()
            )
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON schema")
    except Exception as e:
        logger.error(f"Upload and extract failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-large-document", response_model=ExtractionResponse)
async def extract_large_document(
    request: LargeDocumentRequest,
    background_tasks: BackgroundTasks,
    system: StructuredExtractionSystem = Depends(get_extraction_system)
):
    """
    Extract structured data from large documents using splitting strategy.
    
    This endpoint first splits the document into sections, then extracts from each section.
    Useful for very large documents (50+ pages) or complex multi-section documents.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job in storage
    job_storage[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(),
        "request": request.dict()
    }
    
    # Background task for large document processing
    async def process_large_document():
        try:
            job_storage[job_id]["status"] = "processing"
            
            # Process large document with splitting
            results = system.process_large_document_with_splitting(
                document_url=request.document_url,
                schema=request.json_schema,
                split_descriptions=request.split_descriptions,
                system_prompt=request.system_prompt
            )
            
            # Aggregate results from all sections
            aggregated_data = {}
            all_low_confidence_fields = []
            total_processing_time = 0
            
            for section_name, section_result in results.items():
                aggregated_data[section_name] = section_result.data
                all_low_confidence_fields.extend(
                    [f"{section_name}.{field}" for field in section_result.low_confidence_fields or []]
                )
                total_processing_time += section_result.processing_time
            
            # Update job with aggregated results
            job_storage[job_id].update({
                "status": "completed",
                "data": aggregated_data,
                "processing_time": total_processing_time,
                "low_confidence_fields": all_low_confidence_fields,
                "completed_at": datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Large document job {job_id} failed: {str(e)}")
            job_storage[job_id].update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.now()
            })
    
    background_tasks.add_task(process_large_document)
    
    return ExtractionResponse(
        job_id=job_id,
        status="queued",
        created_at=datetime.now()
    )

@app.get("/jobs/{job_id}", response_model=ExtractionResponse)
async def get_job_status(job_id: str):
    """
    Get the status and results of an extraction job.
    
    Use this endpoint to check the progress of async extraction jobs
    and retrieve results when completed.
    """
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = job_storage[job_id]
    return ExtractionResponse(**job_data)

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed or failed job from storage."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_storage[job_id]
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/jobs", response_model=List[ExtractionResponse])
async def list_jobs(status: Optional[str] = None, limit: int = 100):
    """
    List all jobs, optionally filtered by status.
    
    Args:
        status: Filter by job status (queued, processing, completed, failed)
        limit: Maximum number of jobs to return
    """
    jobs = list(job_storage.values())
    
    if status:
        jobs = [job for job in jobs if job.get("status") == status]
    
    # Sort by creation time (newest first) and limit
    jobs.sort(key=lambda x: x.get("created_at", datetime.min), reverse=True)
    jobs = jobs[:limit]
    
    return [ExtractionResponse(**job) for job in jobs]

@app.post("/analyze-complexity", response_model=ComplexityAnalysisResponse)
async def analyze_complexity(
    request: ComplexityAnalysisRequest,
    system: StructuredExtractionSystem = Depends(get_extraction_system)
):
    """
    Analyze schema and optional document complexity without performing extraction.
    
    Useful for understanding processing requirements and costs before extraction.
    """
    try:
        # Analyze schema complexity
        schema_complexity = system._analyze_schema_complexity(request.json_schema)
        
        # Analyze document complexity if URL provided
        doc_complexity = None
        if request.document_url:
            doc_complexity = system._analyze_document_complexity(request.document_url)
        
        # Determine recommended strategy
        overall_complexity = doc_complexity or schema_complexity
        recommended_strategy = system._determine_processing_strategy(schema_complexity, overall_complexity)
        
        return ComplexityAnalysisResponse(
            schema_complexity=schema_complexity.value,
            document_complexity=doc_complexity.value if doc_complexity else None,
            recommended_strategy={
                "ocr_mode": recommended_strategy.ocr_mode,
                "array_extract": recommended_strategy.array_extract,
                "model_preference": recommended_strategy.model_preference,
                "generate_citations": recommended_strategy.generate_citations,
                "use_async": recommended_strategy.use_async,
                "estimated_cost_multiplier": recommended_strategy.estimated_cost_multiplier
            },
            estimated_cost_multiplier=recommended_strategy.estimated_cost_multiplier,
            analysis_details={
                "schema_analysis": "Complex schema analysis details would go here",
                "document_analysis": "Document analysis details would go here" if request.document_url else None
            }
        )
        
    except Exception as e:
        logger.error(f"Complexity analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints for testing

@app.get("/sample-schema")
async def get_sample_schema():
    """Get a sample complex schema for testing."""
    from utils.extraction_pipeline import create_complex_schema
    return create_complex_schema()

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Structured Extraction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health"
    }

@app.get("/debug/test-extraction")
async def debug_test_extraction(system: StructuredExtractionSystem = Depends(get_extraction_system)):
    """Debug endpoint to test extraction with a simple example."""
    try:
        # Simple test schema
        test_schema = {
            "type": "object",
            "properties": {
                "test_field": {"type": "string", "description": "A test field"}
            }
        }
        
        # Mock extraction result for testing
        from utils.extraction_pipeline import ExtractionResult, ProcessingStrategy
        
        mock_result = ExtractionResult(
            data={"test_field": "test_value"},
            confidence_scores={"test_field": 0.95},
            processing_time=1.0,
            strategy_used=ProcessingStrategy(
                ocr_mode="standard",
                array_extract=False,
                model_preference="fast",
                generate_citations=False,
                use_async=False,
                estimated_cost_multiplier=1.0
            ),
            citations=[],
            low_confidence_fields=[]
        )
        
        # Check data type
        data_type = type(mock_result.data).__name__
        
        return {
            "message": "Debug test extraction",
            "mock_data": mock_result.data,
            "data_type": data_type,
            "is_dict": isinstance(mock_result.data, dict),
            "is_list": isinstance(mock_result.data, list),
            "system_available": True
        }
        
    except Exception as e:
        return {
            "message": "Debug test failed",
            "error": str(e),
            "system_available": False
        }

if __name__ == "__main__":
    import uvicorn
    
    # Configuration for development
    uvicorn.run(
        "main:app",  # Replace with your module name
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )