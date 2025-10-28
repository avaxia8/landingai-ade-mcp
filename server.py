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
import hashlib
from collections import defaultdict

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
API_KEY = os.environ.get("LANDINGAI_API_KEY")
if not API_KEY:
    logger.warning("No API key found. Set LANDINGAI_API_KEY environment variable.")

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

# ============= Internal API Functions =============
# These functions contain the actual API logic and can be called by both MCP tools and process_folder

async def _parse_document_internal(
    document_path: Optional[str] = None,
    document_url: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None
) -> Dict[str, Any]:
    """Internal function for document parsing - contains the actual API logic"""
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
            "error": "API key not configured. Set LANDINGAI_API_KEY environment variable."
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

async def _extract_data_internal(
    schema: Union[Dict[str, Any], str],
    markdown: Optional[str] = None,
    markdown_url: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Internal function for data extraction - contains the actual API logic"""
    logger.info("Extracting data with provided schema")
    
    # Parse schema if it's a string
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Invalid JSON schema: {str(e)}"
            }
    
    if not schema:
        return {
            "status": "error",
            "error": "schema is required"
        }
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY environment variable."
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

async def _create_parse_job_internal(
    document_path: Optional[str] = None,
    document_url: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    output_save_url: Optional[str] = None
) -> Dict[str, Any]:
    """Internal function for creating parse jobs - contains the actual API logic"""
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
            "error": "API key not configured. Set LANDINGAI_API_KEY environment variable."
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
    return await _parse_document_internal(document_path, document_url, model, split)


@mcp.tool()
async def extract_data(
    schema: Union[Dict[str, Any], str],
    markdown: Optional[str] = None,
    markdown_url: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extract structured data from markdown content using a JSON schema.
    
    The schema determines what key-value pairs are extracted from the markdown.
    
    Args:
        schema: JSON schema dict or string for field extraction (required). Defines what to extract.
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
    return await _extract_data_internal(schema, markdown, markdown_url, model)

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
    return await _create_parse_job_internal(document_path, document_url, model, split, output_save_url)

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
            "error": "API key not configured. Set LANDINGAI_API_KEY environment variable."
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
                    logger.info(f"Output URL found: {output_url}")
                    print(f"📎 Output URL: {output_url}")
                    result["_message"] = f"Job completed. Output URL: {output_url}"
                    
                    # Try to fetch the results from the pre-signed URL
                    try:
                        logger.info(f"Fetching results from output URL...")
                        print(f"📥 Downloading from URL...")
                        
                        # Use a new client without auth headers for the pre-signed URL
                        async with httpx.AsyncClient(timeout=60.0) as fetch_client:
                            fetch_response = await fetch_client.get(output_url)
                            
                            if fetch_response.status_code == 200:
                                # Parse the fetched data
                                fetched_data = fetch_response.json()
                                
                                # Since we're fetching from output_url, this is a large file
                                # Save it to disk to avoid context window issues
                                if isinstance(fetched_data, dict) and "markdown" in fetched_data:
                                    markdown_content = fetched_data["markdown"]
                                    markdown_length = len(markdown_content)
                                    
                                    from datetime import datetime
                                    
                                    # Create output filename with timestamp
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    output_file = f"/tmp/{job_id}_{timestamp}_output.md"
                                    
                                    # Write markdown to file
                                    with open(output_file, 'w', encoding='utf-8') as f:
                                        f.write(markdown_content)
                                    
                                    file_size_mb = markdown_length / (1024 * 1024)
                                    logger.info(f"Large result saved to: {output_file} ({file_size_mb:.2f} MB)")
                                    print(f"💾 Large result saved to: {output_file} ({file_size_mb:.2f} MB)")
                                    
                                    # Return file info instead of full content
                                    result["data_file"] = output_file
                                    result["_message"] = f"Job completed. Large results ({file_size_mb:.2f} MB) saved to: {output_file}"
                                    result["_fetch_status"] = "success_saved_to_file"
                                    result["_summary"] = {
                                        "markdown_length": markdown_length,
                                        "chunks_count": len(fetched_data.get("chunks", [])),
                                        "has_splits": "splits" in fetched_data and bool(fetched_data.get("splits")),
                                        "file_path": output_file,
                                        "file_size_mb": file_size_mb
                                    }
                                    
                                    # Include only metadata and structure info, not the large content
                                    result["metadata"] = fetched_data.get("metadata", {})
                                    
                                    # Include a preview of the content (first 1000 chars)
                                    result["preview"] = markdown_content[:1000] + "..." if markdown_length > 1000 else markdown_content
                                else:
                                    # No markdown or not a dict, include as-is
                                    result["data"] = fetched_data
                                    result["_message"] = "Job completed. Results fetched from output URL successfully."
                                    result["_fetch_status"] = "success"
                            else:
                                logger.warning(f"Failed to fetch from output URL: {fetch_response.status_code}")
                                result["_message"] = f"Job completed. Results at output_url (fetch failed: {fetch_response.status_code})"
                                result["_fetch_status"] = f"failed_{fetch_response.status_code}"
                                result["_fetch_note"] = "Use download_from_url tool with the output_url to retry"
                                
                    except Exception as e:
                        logger.error(f"Error fetching from output URL: {e}")
                        result["_message"] = "Job completed. Results at output_url (auto-fetch failed)"
                        result["_fetch_status"] = "error"
                        result["_fetch_error"] = str(e)
                        result["_fetch_note"] = "Use download_from_url tool with the output_url to retry"
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
            "error": "API key not configured. Set LANDINGAI_API_KEY environment variable."
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
async def download_from_url(url: str, save_to_file: bool = True) -> Dict[str, Any]:
    """
    Download results from a pre-signed URL (typically from completed parse jobs).
    
    This is useful when you have an output_url from a completed job and want to 
    fetch the results separately, or if auto-fetch failed.
    
    Args:
        url: The pre-signed URL to download from (usually from job output_url)
        save_to_file: If True, save large results to file instead of memory (default: True)
    
    Returns:
        The downloaded data or file path (for large results)
    """
    logger.info(f"Downloading from URL: {url}")
    print(f"📎 URL: {url}")
    print(f"📥 Downloading...")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Don't use auth headers for pre-signed URLs
            response = await client.get(url)
            
            if response.status_code == 200:
                try:
                    # Try to parse as JSON
                    data = response.json()
                    
                    # Save markdown to file (download_from_url is typically used for large files)
                    if save_to_file and isinstance(data, dict) and "markdown" in data:
                        markdown_content = data["markdown"]
                        markdown_length = len(markdown_content)
                        
                        # Always save to file when using download_from_url (indicates large file)
                        if True:  # Always save since this tool is for large files from output_url
                            from datetime import datetime
                            import hashlib
                            
                            # Create a simple hash from URL for consistent naming
                            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_file = f"/tmp/download_{url_hash}_{timestamp}.md"
                            
                            # Write markdown to file
                            with open(output_file, 'w', encoding='utf-8') as f:
                                f.write(markdown_content)
                            
                            file_size_mb = markdown_length / (1024 * 1024)
                            logger.info(f"Large result saved to: {output_file} ({file_size_mb:.2f} MB)")
                            print(f"💾 Large result saved to: {output_file} ({file_size_mb:.2f} MB)")
                            
                            # Return file info instead of full content
                            result = {
                                "status": "success",
                                "data_file": output_file,
                                "_message": f"Large results ({file_size_mb:.2f} MB) saved to: {output_file}",
                                "_summary": {
                                    "markdown_length": markdown_length,
                                    "chunks_count": len(data.get("chunks", [])),
                                    "has_metadata": "metadata" in data,
                                    "file_path": output_file,
                                    "file_size_mb": file_size_mb
                                },
                                "metadata": data.get("metadata", {}),
                                "preview": markdown_content[:1000] + "..." if markdown_length > 1000 else markdown_content
                            }
                            
                            return result
                    
                    # Small enough or not markdown - include in response
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

# Note: process_folder will call the existing parse_document and extract_data functions directly
# since they are just async functions, not MCP tool decorators that prevent direct calls

@mcp.tool()
async def process_folder(
    folder_path: str,
    operation: str = "parse",
    schema: Optional[Union[Dict[str, Any], str]] = None,
    file_types: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
    max_concurrent: int = 3,
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Process all supported files in a folder - parse documents or extract structured data.
    
    Supported formats:
    - Images: APNG, BMP, DCX, DDS, DIB, GD, GIF, ICNS, JP2, JPEG, JPG, PCX, PNG, PPM, PSD, TGA, TIFF, WEBP
    - Documents: PDF, DOC, DOCX, PPT, PPTX, ODP, ODT
    
    Args:
        folder_path: Path to folder containing documents
        operation: "parse" (extract text/tables) or "extract" (extract structured data)
        schema: JSON schema dict or string for extraction (required if operation="extract")
        file_types: Comma-separated file extensions to process (e.g., "pdf,jpg"). None = all supported
        model: Model version to use (optional, e.g., "dpt-2-latest")
        split: Set to "page" to split documents by pages (optional)
        max_concurrent: Maximum number of concurrent operations (default: 3)
        save_results: Save results to files in ade_results folder (default: True)
    
    Returns:
        Processing summary with file results, paths, and aggregated data (for extract)
    
    Examples:
        # Parse all PDFs
        await process_folder("/path/to/docs", operation="parse", file_types="pdf")
        
        # Extract invoice data from all documents
        schema = {"type": "object", "properties": {"invoice_no": {"type": "string"}}}
        await process_folder("/path/to/invoices", operation="extract", schema=schema)
    """
    # Validate inputs
    if operation not in ["parse", "extract"]:
        return {
            "status": "error",
            "error": "Operation must be 'parse' or 'extract'"
        }
    
    if operation == "extract" and not schema:
        return {
            "status": "error",
            "error": "Schema is required for extract operation"
        }
    
    # Parse schema if it's a string
    if schema and isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "error": f"Invalid JSON schema: {str(e)}"
            }
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return {
            "status": "error",
            "error": f"Folder not found or not a directory: {folder_path}"
        }
    
    if not API_KEY:
        return {
            "status": "error",
            "error": "API key not configured. Set LANDINGAI_API_KEY environment variable."
        }
    
    # Define supported extensions
    SUPPORTED_EXTENSIONS = {
        # Images
        '.apng', '.bmp', '.dcx', '.dds', '.dib', '.gd', '.gif', 
        '.icns', '.jp2', '.jpeg', '.jpg', '.pcx', '.png', '.ppm', 
        '.psd', '.tga', '.tiff', '.tif', '.webp',
        # Documents
        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.odp', '.odt'
    }
    
    # Filter extensions if specified
    if file_types:
        requested_exts = {f".{ext.lower().strip()}" for ext in file_types.split(",")}
        allowed_exts = requested_exts & SUPPORTED_EXTENSIONS
    else:
        allowed_exts = SUPPORTED_EXTENSIONS
    
    # Find all matching files
    all_files = []
    for ext in allowed_exts:
        all_files.extend(folder.glob(f"*{ext}"))
        all_files.extend(folder.glob(f"*{ext.upper()}"))
    
    # Remove duplicates and sort
    all_files = sorted(set(all_files))
    
    if not all_files:
        return {
            "status": "success",
            "message": f"No supported files found in {folder_path}",
            "file_types_searched": list(allowed_exts),
            "total_files": 0
        }
    
    logger.info(f"Found {len(all_files)} files to process in {folder_path}")
    
    # Prepare output directory if saving results
    output_dir = None
    if save_results:
        output_dir = folder / "ade_results"
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Results will be saved to: {output_dir}")
    
    # Track processing results
    processed_files = []
    failed_files = []
    aggregated_data = [] if operation == "extract" else None
    start_time = datetime.now()
    
    # Group files by size for optimal processing
    small_files = []  # < 10MB
    large_files = []  # >= 10MB
    
    for file_path in all_files:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb < 10:
            small_files.append(file_path)
        else:
            large_files.append(file_path)
    
    logger.info(f"Processing {len(small_files)} small files (<10MB) and {len(large_files)} large files (>=10MB)")
    
    # Process small files directly in batches
    for i in range(0, len(small_files), max_concurrent):
        batch = small_files[i:i + max_concurrent]
        tasks = []
        
        for file_path in batch:
            if operation == "parse":
                tasks.append(_parse_document_internal(document_path=str(file_path), model=model, split=split))
            else:  # extract
                # For extraction, we need to parse first then extract
                async def parse_and_extract(fp=file_path, s=schema, m=model, sp=split):
                    parse_result = await _parse_document_internal(document_path=str(fp), model=m, split=sp)
                    if parse_result.get("status") == "error":
                        return parse_result
                    markdown = parse_result.get("markdown", "")
                    if not markdown:
                        return {"status": "error", "error": "No markdown content to extract from"}
                    extract_result = await _extract_data_internal(schema=s, markdown=markdown)
                    if extract_result.get("status") == "success":
                        return {
                            "status": "success",
                            "extraction": extract_result.get("extraction"),
                            "markdown": markdown,
                            "metadata": parse_result.get("metadata")
                        }
                    return extract_result
                tasks.append(parse_and_extract())
        
        # Execute batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for file_path, result in zip(batch, results):
            if isinstance(result, Exception):
                failed_files.append({
                    "filename": file_path.name,
                    "error": str(result)
                })
                logger.error(f"Failed to process {file_path.name}: {result}")
            elif result.get("status") == "error":
                failed_files.append({
                    "filename": file_path.name,
                    "error": result.get("error", "Unknown error")
                })
            else:
                # Save results if requested
                output_path = None
                if save_results and output_dir:
                    file_output_dir = output_dir / f"{file_path.stem}_{operation}"
                    file_output_dir.mkdir(exist_ok=True)
                    output_path = str(file_output_dir)
                    
                    if operation == "parse":
                        # Save markdown content
                        if "markdown" in result:
                            (file_output_dir / "content.md").write_text(
                                result["markdown"], encoding="utf-8"
                            )
                        # Save metadata
                        if "metadata" in result:
                            (file_output_dir / "metadata.json").write_text(
                                json.dumps(result["metadata"], indent=2), encoding="utf-8"
                            )
                        # Save chunks if present
                        if "chunks" in result and result["chunks"]:
                            (file_output_dir / "chunks.json").write_text(
                                json.dumps(result["chunks"], indent=2), encoding="utf-8"
                            )
                    else:  # extract
                        # Save extracted data
                        if "extraction" in result:
                            (file_output_dir / "data.json").write_text(
                                json.dumps(result["extraction"], indent=2), encoding="utf-8"
                            )
                            # Add to aggregated data
                            aggregated_entry = {
                                "source_file": file_path.name,
                                **result["extraction"]
                            }
                            aggregated_data.append(aggregated_entry)
                        # Save source markdown
                        if "markdown" in result:
                            (file_output_dir / "source.md").write_text(
                                result["markdown"], encoding="utf-8"
                            )
                
                processed_files.append({
                    "filename": file_path.name,
                    "status": "success",
                    "pages": result.get("metadata", {}).get("page_count", 0) if operation == "parse" else None,
                    "has_data": bool(result.get("extraction")) if operation == "extract" else None,
                    "output_path": output_path
                })
    
    # Process large files using parse jobs
    if large_files:
        logger.info(f"Creating parse jobs for {len(large_files)} large files")
        jobs = []
        
        # Create jobs for large files
        for file_path in large_files:
            try:
                job_result = await _create_parse_job_internal(
                    document_path=str(file_path),
                    model=model,
                    split=split
                )
                if job_result.get("status") == "success":
                    jobs.append({
                        "file_path": file_path,
                        "job_id": job_result["job_id"]
                    })
                else:
                    failed_files.append({
                        "filename": file_path.name,
                        "error": job_result.get("error", "Failed to create parse job")
                    })
            except Exception as e:
                failed_files.append({
                    "filename": file_path.name,
                    "error": str(e)
                })
        
        # Monitor jobs until completion
        if jobs:
            logger.info(f"Monitoring {len(jobs)} parse jobs...")
            pending_jobs = jobs.copy()
            
            while pending_jobs:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                for job in pending_jobs[:]:
                    try:
                        status_result = await get_parse_job_status(job["job_id"])
                        
                        if status_result.get("status") == "completed":
                            pending_jobs.remove(job)
                            
                            # Process based on operation
                            if operation == "extract" and status_result.get("data"):
                                # Extract from parsed markdown
                                markdown = status_result["data"].get("markdown", "")
                                extract_result = await _extract_data_internal(
                                    schema=schema,
                                    markdown=markdown
                                )
                                
                                if extract_result.get("status") == "success":
                                    result = {
                                        "extraction": extract_result.get("extraction"),
                                        "markdown": markdown,
                                        "metadata": status_result["data"].get("metadata")
                                    }
                                else:
                                    result = extract_result
                            else:
                                result = status_result.get("data", status_result)
                            
                            # Save results
                            output_path = None
                            if save_results and output_dir:
                                file_output_dir = output_dir / f"{job['file_path'].stem}_{operation}"
                                file_output_dir.mkdir(exist_ok=True)
                                output_path = str(file_output_dir)
                                
                                if operation == "parse" and "data_file" in status_result:
                                    # Large file saved to disk
                                    import shutil
                                    shutil.copy(status_result["data_file"], file_output_dir / "content.md")
                                elif operation == "parse" and result.get("markdown"):
                                    (file_output_dir / "content.md").write_text(
                                        result["markdown"], encoding="utf-8"
                                    )
                                elif operation == "extract" and result.get("extraction"):
                                    (file_output_dir / "data.json").write_text(
                                        json.dumps(result["extraction"], indent=2), encoding="utf-8"
                                    )
                                    aggregated_data.append({
                                        "source_file": job["file_path"].name,
                                        **result["extraction"]
                                    })
                            
                            processed_files.append({
                                "filename": job["file_path"].name,
                                "status": "success",
                                "job_id": job["job_id"],
                                "output_path": output_path
                            })
                            
                        elif status_result.get("status") == "failed":
                            pending_jobs.remove(job)
                            failed_files.append({
                                "filename": job["file_path"].name,
                                "error": status_result.get("failure_reason", "Job failed")
                            })
                    except Exception as e:
                        logger.error(f"Error checking job {job['job_id']}: {e}")
                
                if pending_jobs:
                    logger.info(f"Still waiting for {len(pending_jobs)} jobs to complete...")
    
    # Calculate summary
    processing_time = (datetime.now() - start_time).total_seconds()
    
    # Save summary if requested
    if save_results and output_dir:
        summary = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "folder_path": str(folder_path),
            "processing_time_s": processing_time,
            "total_files": len(all_files),
            "processed": len(processed_files),
            "failed": len(failed_files),
            "processed_files": processed_files,
            "failed_files": failed_files
        }
        
        if aggregated_data:
            summary["aggregated_data"] = aggregated_data
            # Also save aggregated data separately
            (output_dir / "extracted_data.json").write_text(
                json.dumps(aggregated_data, indent=2), encoding="utf-8"
            )
        
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    
    # Prepare response
    result = {
        "status": "success",
        "operation": operation,
        "summary": {
            "total_files": len(all_files),
            "processed": len(processed_files),
            "failed": len(failed_files),
            "processing_time_s": round(processing_time, 1)
        },
        "processed_files": processed_files,
        "failed_files": failed_files
    }
    
    if save_results and output_dir:
        result["results_path"] = str(output_dir)
    
    if aggregated_data:
        result["aggregated_data"] = aggregated_data
    
    return result


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
            "process_folder",
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