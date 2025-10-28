"""
LandingAI ADE MCP Server
=====================================================
A Model Context Protocol server that directly integrates with LandingAI's ADE REST API.
"""

import os
import json
import logging
import httpx
import asyncio
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the MCP Server
mcp = FastMCP(
    "landingai-ade-mcp",
    instructions="Model Context Protocol server for LandingAI's Agentic Document Extraction API. Direct REST API integration for document parsing, data extraction, and async job management.",
    version="2.0.0"
)

# Configuration
API_BASE_URL = "https://api.va.landing.ai"
API_BASE_URL_EU = "https://api.va.eu-west-1.landing.ai"

# Get API key from environment
API_KEY = os.environ.get("LANDINGAI_API_KEY", os.environ.get("VISION_AGENT_API_KEY"))
if not API_KEY:
    logger.warning("No API key found. Set LANDINGAI_API_KEY or VISION_AGENT_API_KEY environment variable.")

# ============= Helper Functions =============

def get_headers() -> Dict[str, str]:
    """Get authorization headers for API requests"""
    if not API_KEY:
        raise ValueError("API key not configured")
    return {
        "Authorization": f"Bearer {API_KEY}"
    }

async def handle_api_response(response: httpx.Response, add_status: bool = True) -> Dict[str, Any]:
    """Handle API response and errors"""
    try:
        if response.status_code == 200:
            result = response.json()
            if add_status:
                return {"status": "success", **result}
            return result
        elif response.status_code == 202:
            # Parse job created
            result = response.json()
            if add_status:
                return {"status": "accepted", **result}
            return result
        elif response.status_code == 401:
            return {
                "status": "error",
                "error": "Authentication failed. Check your API key.",
                "status_code": 401
            }
        elif response.status_code == 403:
            return {
                "status": "error",
                "error": "Access forbidden. Check your permissions.",
                "status_code": 403
            }
        elif response.status_code == 404:
            return {
                "status": "error",
                "error": "Resource not found.",
                "status_code": 404
            }
        elif response.status_code == 413:
            return {
                "status": "error",
                "error": "File too large. Use parse job for files over 50MB.",
                "status_code": 413
            }
        elif response.status_code == 429:
            return {
                "status": "error",
                "error": "Rate limit exceeded. Please wait and try again.",
                "status_code": 429
            }
        elif response.status_code == 422:
            error_detail = response.json() if response.text else "Validation error"
            return {
                "status": "error",
                "error": f"Validation error: {error_detail}",
                "status_code": 422
            }
        else:
            return {
                "status": "error",
                "error": f"API error: {response.status_code} - {response.text}",
                "status_code": response.status_code
            }
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": f"Invalid JSON response: {response.text}",
            "status_code": response.status_code
        }

# ============= MCP Tools =============

@mcp.tool()
async def parse_document(
    document_path: Optional[str] = None,
    document_url: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a document (PDF and images) and extract its content.
    
    Supported formats: APNG, BMP, DCX, DDS, DIB, DOC, DOCX, GD, GIF, ICNS, JP2 (JP2000), JPEG, JPG, ODP, ODT, PCX, PDF, PNG, PPT, PPTX, PPM, PSD, TGA, TIFF, WEBP
    See full list: https://docs.landing.ai/ade/ade-file-types
    
    Args:
        document_path: Path to local document file (provide this OR document_url)
        document_url: URL of document to parse (provide this OR document_path)
        model: Model version to use for parsing (optional, e.g., "dpt-2-latest")
        split: Set to "page" to split document by pages (optional)
    
    Returns:
        Response containing:
        - markdown: Full document as markdown text
        - chunks: Array of document chunks with markdown, type, id, and grounding
        - splits: Page/section splits if requested (with page numbers and content)
        - grounding: Location information mapping text to coordinates
        - metadata: Processing details (filename, page_count, duration_ms, credit_usage, etc.)
    """
    if document_path:
        logger.info(f"Parsing document: {document_path}")
    elif document_url:
        logger.info(f"Parsing document from URL: {document_url}")
    else:
        return {
            "status": "error",
            "error": "Must provide either document_path or document_url"
        }
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY or VISION_AGENT_API_KEY environment variable."
        }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            if document_path:
                # Check if file exists
                path = Path(document_path)
                if not path.exists():
                    return {
                        "status": "error",
                        "error": f"File not found: {document_path}"
                    }
                
                # Check file size
                file_size_mb = path.stat().st_size / (1024 * 1024)
                if file_size_mb > 50:
                    logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Consider using create_parse_job for better performance.")
                
                # Prepare multipart form data
                with open(path, 'rb') as f:
                    files = {'document': (path.name, f, 'application/octet-stream')}
                    data = {}
                    if model:
                        data['model'] = model
                    if split:
                        data['split'] = split
                    
                    response = await client.post(
                        f"{API_BASE_URL}/v1/ade/parse",
                        headers=get_headers(),
                        files=files,
                        data=data if data else None
                    )
            else:  # document_url
                # Prepare form data with URL
                data = {'document_url': document_url}
                if model:
                    data['model'] = model
                if split:
                    data['split'] = split
                
                response = await client.post(
                    f"{API_BASE_URL}/v1/ade/parse",
                    headers=get_headers(),
                    data=data
                )
        
        result = await handle_api_response(response)
        if result.get("status") == "success":
            logger.info("Document parsed successfully")
            # Add helpful summary info
            if "chunks" in result:
                result["chunks_count"] = len(result["chunks"])
            if "metadata" in result and "page_count" in result["metadata"]:
                result["page_count"] = result["metadata"]["page_count"]
            if "markdown" in result:
                result["markdown_length"] = len(result["markdown"])
        return result
        
    except FileNotFoundError:
        return {
            "status": "error",
            "error": f"File not found: {document_path}"
        }
    except PermissionError:
        return {
            "status": "error",
            "error": f"Permission denied accessing file: {document_path}"
        }
    except httpx.TimeoutException:
        return {
            "status": "error",
            "error": "Request timeout. File may be too large - consider using async job."
        }
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


@mcp.tool()
async def extract_data(
    schema: Dict[str, Any],
    markdown: Optional[str] = None,
    markdown_url: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract structured data from markdown content using a JSON schema.
    
    The schema determines what key-value pairs are extracted from the markdown.
    
    Args:
        schema: JSON schema for field extraction (required). Defines what to extract.
                Must be a valid JSON object with type definitions.
        markdown: The markdown file path or markdown content string (provide this OR markdown_url)
        markdown_url: URL to markdown file (provide this OR markdown)
        model: Model version for extraction (optional, e.g., "extract-latest" for latest version)
    
    Returns:
        Response with exactly these fields:
        - extraction: Object containing the extracted key-value pairs matching your schema
        - extraction_metadata: Object with extracted pairs and chunk references for each value
        - metadata: Object containing:
            - filename: Name of processed file
            - org_id: Organization ID
            - duration_ms: Processing time in milliseconds
            - credit_usage: Credits consumed
            - job_id: Unique job identifier
            - version: API version used
    """
    logger.info("Extracting data with provided schema")
    
    if not schema:
        return {
            "status": "error",
            "error": "schema is required"
        }
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY or VISION_AGENT_API_KEY environment variable."
        }
    
    try:
        # Validate schema
        if not isinstance(schema, dict):
            return {
                "status": "error",
                "error": "Schema must be a valid JSON object (dict)"
            }
        
        # Ensure schema is properly formatted
        if "type" not in schema:
            logger.warning("Schema missing 'type' field, adding 'object' as default")
            schema = {"type": "object", "properties": schema}
        
        # Convert schema to JSON string
        schema_json = json.dumps(schema)
        
        # Validate input sources
        if markdown is None and markdown_url is None:
            return {
                "status": "error",
                "error": "Must provide either markdown (file path or content) or markdown_url"
            }
        
        if markdown is not None and markdown_url is not None:
            return {
                "status": "error",
                "error": "Provide only ONE of: markdown or markdown_url"
            }
        
        # Prepare and send request based on input type
        async with httpx.AsyncClient(timeout=60.0) as client:
            if markdown is not None:
                # Check if markdown is a file path or content string
                is_file = False
                path = None
                
                # Try to interpret as file path
                try:
                    # Handle both Path objects and strings
                    if isinstance(markdown, Path):
                        path = markdown
                    else:
                        path = Path(markdown)
                    
                    # Check if it's an existing file
                    if path.exists() and path.is_file():
                        is_file = True
                except:
                    # Not a valid path, treat as content
                    is_file = False
                
                if is_file:
                    # Markdown file path provided
                    logger.info(f"Extracting from markdown file: {path}")
                    with open(path, 'rb') as f:
                        files = {
                            'schema': (None, schema_json, 'application/json'),
                            'markdown': (path.name, f, 'text/markdown')
                        }
                        data = {}
                        if model:
                            data['model'] = model
                        
                        response = await client.post(
                            f"{API_BASE_URL}/v1/ade/extract",
                            headers=get_headers(),
                            files=files,
                            data=data if data else None
                        )
                else:
                    # Markdown content provided as string
                    logger.info("Extracting from markdown content string")
                    files = {
                        'schema': (None, schema_json, 'application/json'),
                        'markdown': ('content.md', str(markdown).encode('utf-8'), 'text/markdown')
                    }
                    data = {}
                    if model:
                        data['model'] = model
                    
                    response = await client.post(
                        f"{API_BASE_URL}/v1/ade/extract",
                        headers=get_headers(),
                        files=files,
                        data=data if data else None
                    )
                    
            else:  # markdown_url is not None
                # Markdown URL provided
                logger.info(f"Extracting from markdown URL: {markdown_url}")
                # When using URL, send as form data (not multipart)
                data = {
                    'schema': schema_json,
                    'markdown_url': markdown_url
                }
                if model:
                    data['model'] = model
                
                response = await client.post(
                    f"{API_BASE_URL}/v1/ade/extract",
                    headers=get_headers(),
                    data=data
                )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            
            # Validate expected response structure
            if all(key in result for key in ['extraction', 'extraction_metadata', 'metadata']):
                logger.info("Data extraction successful")
                
                # Add helpful summary
                extraction = result.get('extraction', {})
                metadata = result.get('metadata', {})
                
                # Add summary info prefixed with underscore
                result['_summary'] = {
                    'fields_extracted': len(extraction),
                    'duration_ms': metadata.get('duration_ms'),
                    'credit_usage': metadata.get('credit_usage'),
                    'extracted_keys': list(extraction.keys())
                }
                
                # Add success status for consistency
                result['status'] = 'success'
            else:
                logger.warning("Response missing expected fields")
                result['status'] = 'success'  # Still successful API call
                
            return result
            
        elif response.status_code == 206:
            # Partial success - some schema validation issues
            result = response.json()
            result['status'] = 'partial'
            result['_message'] = 'Extraction partially successful with schema validation warnings'
            return result
            
        else:
            # Handle error responses
            result = await handle_api_response(response)
            return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in schema: {e}")
        return {
            "status": "error",
            "error": f"Invalid JSON schema: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def create_parse_job(
    document_path: Optional[str] = None,
    document_url: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    output_save_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a parse job for large documents (asynchronous processing).
    
    Use this for documents over 50MB or when you need non-blocking processing.
    Provide either document_path (local file) or document_url.
    
    Supported formats: APNG, BMP, DCX, DDS, DIB, DOC, DOCX, GD, GIF, ICNS, JP2 (JP2000), JPEG, JPG, ODP, ODT, PCX, PDF, PNG, PPT, PPTX, PPM, PSD, TGA, TIFF, WEBP
    See full list: https://docs.landing.ai/ade/ade-file-types
    
    Args:
        document_path: Path to local document file (provide this OR document_url)
        document_url: URL of document to parse (provide this OR document_url)
        model: Model version to use for parsing (optional, e.g., "dpt-2-latest")
        split: Set to "page" to split document by pages (optional)
        output_save_url: URL to save output for Zero Data Retention (optional)
    
    Returns:
        Response with:
        - job_id: Unique identifier for tracking the parse job
        - status: "success" if job created
        - message: Helpful status message
    """
    if document_path:
        logger.info(f"Creating parse job for file: {document_path}")
    elif document_url:
        logger.info(f"Creating parse job for URL: {document_url}")
    else:
        return {
            "status": "error",
            "error": "Must provide either document_path or document_url"
        }
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY or VISION_AGENT_API_KEY environment variable."
        }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if document_path:
                path = Path(document_path)
                if not path.exists():
                    return {
                        "status": "error",
                        "error": f"File not found: {document_path}"
                    }
                
                # Log file size for user awareness
                file_size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"File size: {file_size_mb:.2f} MB")
                
                with open(path, 'rb') as f:
                    files = {'document': (path.name, f, 'application/octet-stream')}
                    
                    # Build data dict with only provided parameters
                    data = {}
                    if model:
                        data['model'] = model
                    if split:
                        data['split'] = split
                    if output_save_url:
                        data['output_save_url'] = output_save_url
                    
                    response = await client.post(
                        f"{API_BASE_URL}/v1/ade/parse/jobs",
                        headers=get_headers(),
                        files=files,
                        data=data if data else None
                    )
            else:  # document_url
                # Build data dict with URL and optional parameters
                data = {'document_url': document_url}
                if model:
                    data['model'] = model
                if split:
                    data['split'] = split
                if output_save_url:
                    data['output_save_url'] = output_save_url
                
                response = await client.post(
                    f"{API_BASE_URL}/v1/ade/parse/jobs",
                    headers=get_headers(),
                    data=data
                )
        
        # Handle 202 Accepted response
        if response.status_code == 202:
            result = response.json()
            job_id = result.get('job_id')
            logger.info(f"Parse job created successfully: {job_id}")
            return {
                "status": "success",
                "job_id": job_id,
                "message": "Parse job created. Use get_parse_job_status to check progress.",
                "note": "For Zero Data Retention, results will be saved to output_save_url if provided."
            }
        else:
            # Handle error responses
            result = await handle_api_response(response)
            return result
        
    except Exception as e:
        logger.error(f"Error creating parse job: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def get_parse_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the status and results of a parse job.
    
    Args:
        job_id: The unique job identifier to check
    
    Returns:
        Complete job information including:
        - job_id: The job identifier
        - status: Current status (pending, processing, completed, failed, cancelled)
        - received_at: Timestamp when job was received
        - progress: Progress from 0.0 to 1.0
        - org_id: Organization ID (optional)
        - version: API version (optional)
        - data: Parse results when completed (null if not complete or using output_url)
            - markdown: Full document as markdown text
            - chunks: Array with markdown, type, id, and grounding per chunk
            - splits: Page/section splits if requested
            - grounding: Location information
            - metadata: Processing details
        - output_url: URL to download results if >1MB or output_save_url was used (optional)
        - metadata: Job processing metadata (optional)
        - failure_reason: Error message if job failed (optional)
    """
    logger.info(f"Getting status for job: {job_id}")
    
    if not job_id:
        return {
            "status": "error",
            "error": "job_id is required"
        }
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY or VISION_AGENT_API_KEY environment variable."
        }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{API_BASE_URL}/v1/ade/parse/jobs/{job_id}",
                headers=get_headers()
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Add helper information for better user experience
            job_status = result.get("status", "unknown")
            progress = result.get("progress", 0)
            
            # Add human-readable message based on status
            if job_status == "completed":
                if "data" in result and result["data"]:
                    # Data is included in response
                    result["_message"] = "Job completed successfully. Results available in 'data' field."
                    if "markdown" in result["data"]:
                        result["_summary"] = {
                            "markdown_length": len(result["data"]["markdown"]),
                            "chunks_count": len(result["data"].get("chunks", [])),
                            "has_splits": "splits" in result["data"] and bool(result["data"]["splits"])
                        }
                elif "output_url" in result and result["output_url"]:
                    # Results available via URL (>1MB or Zero Data Retention)
                    output_url = result["output_url"]
                    result["_message"] = "Job completed. Fetching results from output URL..."
                    
                    # Try to fetch the results from the pre-signed URL
                    try:
                        logger.info(f"Fetching results from output URL: {output_url}")
                        # Use a new client without auth headers for the pre-signed URL
                        async with httpx.AsyncClient(timeout=60.0) as fetch_client:
                            fetch_response = await fetch_client.get(output_url)
                            
                            if fetch_response.status_code == 200:
                                # Parse the fetched data
                                fetched_data = fetch_response.json()
                                
                                # Add the fetched data to the result
                                result["data"] = fetched_data
                                result["_message"] = "Job completed. Results fetched from output URL successfully."
                                result["_fetch_status"] = "success"
                                
                                # Add summary if markdown is present
                                if isinstance(fetched_data, dict) and "markdown" in fetched_data:
                                    result["_summary"] = {
                                        "markdown_length": len(fetched_data["markdown"]),
                                        "chunks_count": len(fetched_data.get("chunks", [])),
                                        "has_splits": "splits" in fetched_data and bool(fetched_data["splits"])
                                    }
                            else:
                                logger.warning(f"Failed to fetch from output URL: {fetch_response.status_code}")
                                result["_message"] = f"Job completed. Results at output_url (fetch failed: {fetch_response.status_code})"
                                result["_fetch_status"] = f"failed_{fetch_response.status_code}"
                                result["_fetch_note"] = "Use the output_url directly to download results"
                                
                    except Exception as e:
                        logger.error(f"Error fetching from output URL: {e}")
                        result["_message"] = "Job completed. Results at output_url (auto-fetch failed)"
                        result["_fetch_status"] = "error"
                        result["_fetch_error"] = str(e)
                        result["_fetch_note"] = "Use the output_url directly to download results"
                else:
                    result["_message"] = "Job completed but no data available."
                    
            elif job_status == "processing":
                result["_message"] = f"Job in progress: {progress*100:.1f}% complete"
                result["_estimated_wait"] = "Check again in 5-10 seconds"
                
            elif job_status == "pending":
                result["_message"] = "Job is queued and waiting to be processed"
                result["_estimated_wait"] = "Processing should start soon"
                
            elif job_status == "failed":
                failure_reason = result.get("failure_reason", "Unknown error")
                result["_message"] = f"Job failed: {failure_reason}"
                
            elif job_status == "cancelled":
                result["_message"] = "Job was cancelled"
            
            # Add timestamp conversion for readability
            if "received_at" in result and isinstance(result["received_at"], (int, float)):
                try:
                    from datetime import datetime
                    dt = datetime.fromtimestamp(result["received_at"] / 1000)  # Assuming milliseconds
                    result["_received_at_iso"] = dt.isoformat()
                except:
                    pass  # Keep original format if conversion fails
            
            logger.info(f"Job {job_id} status: {job_status} ({progress*100:.1f}% complete)")
            return result
            
        elif response.status_code == 404:
            return {
                "status": "error",
                "error": f"Job not found: {job_id}",
                "status_code": 404
            }
        else:
            # Handle other error responses
            result = await handle_api_response(response, add_status=False)
            return result
        
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def list_parse_jobs(
    page: Optional[int] = None,
    pageSize: Optional[int] = None,
    status: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all parse jobs with optional filtering.
    
    Args:
        page: Page number, 0-indexed (optional, default: 0, min: 0)
        pageSize: Number of items per page (optional, default: 10, range: 1-100)
        status: Filter by job status - one of: cancelled, completed, failed, pending, processing (optional)
    
    Returns:
        Response containing:
        - jobs: Array of job objects, each with:
            - job_id: Unique job identifier
            - status: Current job status
            - received_at: Timestamp when job was received
            - progress: Progress from 0.0 to 1.0
            - failure_reason: Error message if job failed (optional)
        - org_id: Organization ID (optional)
        - has_more: Boolean indicating if more pages are available
    """
    # Set defaults and validate ranges
    actual_page = page if page is not None else 0
    actual_page_size = pageSize if pageSize is not None else 10
    
    # Validate page is non-negative
    if actual_page < 0:
        return {
            "status": "error",
            "error": "Page must be >= 0"
        }
    
    # Validate and cap pageSize
    if actual_page_size < 1:
        return {
            "status": "error",
            "error": "Page size must be >= 1"
        }
    if actual_page_size > 100:
        actual_page_size = 100
        logger.info(f"Page size capped at maximum of 100")
    
    # Validate status filter if provided
    valid_statuses = ["cancelled", "completed", "failed", "pending", "processing"]
    if status and status not in valid_statuses:
        return {
            "status": "error",
            "error": f"Invalid status filter. Must be one of: {', '.join(valid_statuses)}"
        }
    
    logger.info(f"Listing parse jobs (page: {actual_page}, pageSize: {actual_page_size}, status: {status})")
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY or VISION_AGENT_API_KEY environment variable."
        }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Build query parameters
            params = {}
            if page is not None:
                params['page'] = actual_page
            if pageSize is not None:
                params['pageSize'] = actual_page_size
            if status:
                params['status'] = status
            
            response = await client.get(
                f"{API_BASE_URL}/v1/ade/parse/jobs",
                headers=get_headers(),
                params=params if params else None
            )
        
        if response.status_code == 200:
            result = response.json()
            
            # Add status to indicate success
            result["status"] = "success"
            
            # Add helpful summary information
            jobs = result.get("jobs", [])
            logger.info(f"Found {len(jobs)} jobs on page {actual_page}")
            
            # Add pagination info
            result["pagination"] = {
                "current_page": actual_page,
                "page_size": actual_page_size,
                "items_on_page": len(jobs),
                "has_more": result.get("has_more", False)
            }
            
            # Count jobs by status for convenience
            if jobs:
                status_counts = {}
                for job in jobs:
                    job_status = job.get("status", "unknown")
                    status_counts[job_status] = status_counts.get(job_status, 0) + 1
                result["status_summary"] = status_counts
                
                # Add human-readable timestamps if available
                for job in jobs:
                    if "received_at" in job and isinstance(job["received_at"], (int, float)):
                        # Convert timestamp to ISO format for readability
                        try:
                            from datetime import datetime
                            dt = datetime.fromtimestamp(job["received_at"] / 1000)  # Assuming milliseconds
                            job["received_at_iso"] = dt.isoformat()
                        except:
                            pass  # Keep original format if conversion fails
            
            return result
        else:
            # Handle error responses
            result = await handle_api_response(response)
            return result
        
    except Exception as e:
        logger.error(f"Error listing parse jobs: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
async def download_from_url(url: str) -> Dict[str, Any]:
    """
    Download results from a pre-signed URL (typically from completed parse jobs).
    
    This is useful when you have an output_url from a completed job and want to 
    fetch the results separately, or if auto-fetch failed.
    
    Args:
        url: The pre-signed URL to download from (usually from job output_url)
    
    Returns:
        The downloaded data (typically contains markdown, chunks, metadata)
    """
    logger.info(f"Downloading from URL: {url}")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Don't use auth headers for pre-signed URLs
            response = await client.get(url)
            
            if response.status_code == 200:
                try:
                    # Try to parse as JSON
                    data = response.json()
                    
                    # Add summary information
                    result = {
                        "status": "success",
                        "data": data,
                        "_message": "Successfully downloaded and parsed data"
                    }
                    
                    # Add summary if it's parse results
                    if isinstance(data, dict):
                        if "markdown" in data:
                            result["_summary"] = {
                                "markdown_length": len(data["markdown"]),
                                "chunks_count": len(data.get("chunks", [])) if "chunks" in data else 0,
                                "has_metadata": "metadata" in data
                            }
                        result["_keys_found"] = list(data.keys())
                    
                    return result
                    
                except json.JSONDecodeError:
                    # Return raw content if not JSON
                    return {
                        "status": "success",
                        "content": response.text[:10000],  # Limit to first 10k chars
                        "content_type": response.headers.get("content-type", "unknown"),
                        "_message": "Downloaded non-JSON content (showing first 10k chars)",
                        "_full_length": len(response.text)
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Failed to download: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "response": response.text[:500] if response.text else None
                }
                
    except httpx.TimeoutException:
        return {
            "status": "error",
            "error": "Download timeout after 60 seconds"
        }
    except Exception as e:
        logger.error(f"Error downloading from URL: {e}")
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Check the health status of the MCP server and API connectivity.
    
    Returns:
        Server status, configuration details, and API connectivity test
    """
    result = {
        "status": "healthy",
        "server": "LandingAI ADE MCP Server (Direct API)",
        "version": "2.0.0",
        "api_key_configured": bool(API_KEY),
        "api_base_url": API_BASE_URL,
        "available_tools": [
            "parse_document",
            "extract_data",
            "create_parse_job",
            "get_parse_job_status",
            "list_parse_jobs",
            "download_from_url",
            "health_check"
        ],
        "timestamp": datetime.now().isoformat()
    }
    
    # Test API connectivity if key is configured
    if API_KEY:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{API_BASE_URL}/v1/ade/parse/jobs",
                    headers=get_headers(),
                    params={'pageSize': 1}
                )
                if response.status_code == 200:
                    result["api_connectivity"] = "connected"
                elif response.status_code == 401:
                    result["api_connectivity"] = "invalid_api_key"
                else:
                    result["api_connectivity"] = f"error_{response.status_code}"
        except Exception as e:
            result["api_connectivity"] = f"connection_failed: {str(e)}"
    else:
        result["api_connectivity"] = "no_api_key"
    
    return result

# ============= Main Entry Point =============

def main():
    """Main entry point for the MCP server"""
    logger.info("Starting LandingAI ADE MCP Server (Direct API)")
    logger.info("API Key configured: " + ("Yes" if API_KEY else "No"))
    
    # Use FastMCP's built-in run method
    mcp.run()

if __name__ == "__main__":
    main()