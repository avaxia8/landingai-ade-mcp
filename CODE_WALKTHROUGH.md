# Code Walkthrough: LandingAI ADE MCP Server

## Architecture Overview

This MCP (Model Context Protocol) server provides direct integration with LandingAI's ADE (Agentic Document Extraction) REST API. It acts as a bridge between Claude Desktop (or other MCP clients) and the LandingAI API.

### Key Design Principles

1. **Direct API Integration**: Uses `httpx` for HTTP requests instead of SDK dependencies
2. **Memory Management**: Automatically saves large files to disk to prevent context window overflow
3. **Transparent Error Handling**: Provides detailed error messages with fallback options
4. **Pre-signed URL Support**: Handles S3 URLs without authentication headers

### Global Configuration

```python
API_BASE_URL = "https://api.va.landing.ai"
API_KEY = os.environ.get("LANDINGAI_API_KEY")
```

- API key must be set via LANDINGAI_API_KEY environment variable
- All API requests use Bearer token authentication
- Base URL points to LandingAI's Vision Agent API

### Architecture: Internal Functions Pattern

To solve the issue where MCP-decorated functions become FunctionTool objects that can't be called directly, the code uses an internal functions pattern:

```python
# Internal function with actual API logic
async def _parse_document_internal(...):
    # All the API logic here
    
# MCP tool is a thin wrapper
@mcp.tool()
async def parse_document(...):
    return await _parse_document_internal(...)
```

This pattern allows:
- `process_folder` to call internal functions directly
- No code duplication between tools
- Clean separation of API logic from MCP interface

---

## Tool Logic Explanations

### 1. `parse_document` - Document Parsing

**Purpose**: Parse PDFs, images, and office documents to extract text, tables, and visual elements.

#### Input Validation
```python
if document_path:
    # Local file processing
elif document_url:
    # URL processing
else:
    return error  # Must provide one source
```

#### Core Logic Flow

1. **File Size Check** (for local files):
```python
file_size_mb = path.stat().st_size / (1024 * 1024)
if file_size_mb > 50:
    logger.warning("Large file detected. Consider using create_parse_job")
```
- Files > 50MB should use async jobs for better performance

2. **Multipart Form Upload** (for local files):
```python
with open(path, 'rb') as f:
    files = {'document': (path.name, f, 'application/octet-stream')}
    data = {}
    if model: data['model'] = model
    if split: data['split'] = split
```
- File uploaded as multipart form data
- Optional parameters only included if specified

3. **URL Processing**:
```python
data = {'document_url': document_url}
if model: data['model'] = model
if split: data['split'] = split
```
- URL sent as form data, not multipart

#### Response Handling
```python
if result.get("status") == "success":
    # Add helpful summaries
    result["chunks_count"] = len(result["chunks"])
    result["page_count"] = result["metadata"]["page_count"]
    result["markdown_length"] = len(result["markdown"])
```
- Adds summary information for easier consumption
- Returns markdown, chunks, splits, grounding, and metadata

#### Design Decisions
- **Why check file size?** To warn users about potential timeouts
- **Why multipart for files but not URLs?** API expects different content types
- **Why add summaries?** Helps users quickly understand parse results

---

### 2. `extract_data` - Structured Data Extraction

**Purpose**: Extract structured data from markdown using JSON schemas.

#### Parameter Handling
```python
# markdown parameter can be either:
# 1. File path (string or Path object)
# 2. Markdown content string
```

#### Core Logic Flow

1. **Schema Validation**:
```python
if not isinstance(schema, dict):
    return error
if "type" not in schema:
    schema = {"type": "object", "properties": schema}
```
- Ensures schema is valid JSON
- Adds default type if missing

2. **Input Detection**:
```python
# Try to interpret as file path
try:
    path = Path(markdown)
    if path.exists() and path.is_file():
        is_file = True
except:
    is_file = False  # Treat as content string
```
- Intelligently detects if markdown is a file path or content
- Falls back to content if path check fails

3. **File Upload**:
```python
if is_file:
    with open(path, 'rb') as f:
        files = {
            'schema': (None, schema_json, 'application/json'),
            'markdown': (path.name, f, 'text/markdown')
        }
else:
    # Direct content
    files = {
        'schema': (None, schema_json, 'application/json'),
        'markdown': ('content.md', markdown.encode('utf-8'), 'text/markdown')
    }
```

#### Response Structure
```python
# Expected response:
{
    "extraction": {...},        # Extracted key-value pairs
    "extraction_metadata": {...},  # Extraction details
    "metadata": {...}           # Processing metadata
}
```

#### Design Decisions
- **Why auto-detect file vs content?** Better user experience
- **Why validate schema?** Prevents API errors
- **Why use multipart?** API expects schema and markdown as separate parts

---

### 3. `create_parse_job` - Async Job Creation

**Purpose**: Handle large documents (>50MB) with background processing.

#### Core Logic Flow

1. **File Size Logging**:
```python
file_size_mb = path.stat().st_size / (1024 * 1024)
logger.info(f"File size: {file_size_mb:.2f} MB")
```
- Helps users understand why async is needed

2. **Optional Parameters**:
```python
data = {}
if model: data['model'] = model
if split: data['split'] = split
if output_save_url: data['output_save_url'] = output_save_url
```
- Zero Data Retention support via output_save_url
- Only sends parameters that are provided

3. **202 Accepted Response**:
```python
if response.status_code == 202:
    result = response.json()
    job_id = result.get('job_id')
    return {
        "status": "success",
        "job_id": job_id,
        "message": "Parse job created. Use get_parse_job_status to check progress."
    }
```

#### Design Decisions
- **Why log file size?** Transparency about processing requirements
- **Why 202 status?** Standard HTTP code for accepted async operations
- **Why suggest next step?** Guides users through async workflow

---

### 4. `get_parse_job_status` - Job Status & Results Retrieval

**Purpose**: Check job progress and retrieve results, handling large files specially.

#### Core Logic Flow

1. **Status Check**:
```python
job_status = result.get("status", "unknown")
progress = result.get("progress", 0)
```

2. **Three Result Scenarios**:

**A. Small Files (< 1MB) - Inline Data**:
```python
if "data" in result and result["data"]:
    # Data included directly in response
    result["_message"] = "Job completed. Results available in 'data' field."
```

**B. Large Files (> 1MB) - Output URL**:
```python
elif "output_url" in result and result["output_url"]:
    output_url = result["output_url"]
    print(f"📎 Output URL: {output_url}")  # Visibility
    
    # Auto-fetch from S3
    async with httpx.AsyncClient(timeout=60.0) as fetch_client:
        fetch_response = await fetch_client.get(output_url)  # No auth headers!
        
        if fetch_response.status_code == 200:
            fetched_data = fetch_response.json()
            
            # Save to file (it's large!)
            output_file = f"/tmp/{job_id}_{timestamp}_output.md"
            with open(output_file, 'w') as f:
                f.write(fetched_data["markdown"])
            
            # Return path, not content
            result["data_file"] = output_file
            result["preview"] = markdown_content[:1000] + "..."
```

**C. No Data**:
```python
else:
    result["_message"] = "Job completed but no data available."
```

3. **Progress Updates**:
```python
elif job_status == "processing":
    result["_message"] = f"Job in progress: {progress*100:.1f}% complete"
elif job_status == "pending":
    result["_message"] = "Job is queued and waiting to be processed"
elif job_status == "failed":
    result["_message"] = f"Job failed: {failure_reason}"
```

#### Key Design Decisions

1. **Why print output_url?**
   - Transparency - user sees what's happening
   - Debugging - URL available if auto-fetch fails

2. **Why save large files to disk?**
   - Files > 1MB would overflow Claude's context window
   - Disk storage prevents memory issues
   - User gets file path for further processing

3. **Why no auth headers for S3?**
   - Pre-signed URLs include auth in the URL itself
   - Adding headers causes authentication errors
   - Matches standard S3 client behavior

4. **Why include preview?**
   - Shows first 1000 chars for verification
   - Doesn't overflow context
   - Helps identify content

---

### 5. `list_parse_jobs` - Job Listing

**Purpose**: List all parse jobs with filtering and pagination.

#### Parameter Validation
```python
# Defaults and limits
actual_page = page if page is not None else 0
actual_page_size = pageSize if pageSize is not None else 10

# Validation
if actual_page < 0:
    return error("Page must be >= 0")
if actual_page_size < 1:
    return error("Page size must be >= 1")
if actual_page_size > 100:
    actual_page_size = 100  # Cap at API maximum
```

#### Response Enhancement
```python
# Add pagination info
result["pagination"] = {
    "current_page": actual_page,
    "page_size": actual_page_size,
    "items_on_page": len(jobs),
    "has_more": result.get("has_more", False)
}

# Status summary
status_counts = {}
for job in jobs:
    job_status = job.get("status", "unknown")
    status_counts[job_status] = status_counts.get(job_status, 0) + 1
result["status_summary"] = status_counts
```

#### Design Decisions
- **Why validate pagination?** Prevents API errors
- **Why add summaries?** Helps users understand job distribution
- **Why convert timestamps?** Human-readable dates

---

### 6. `process_folder` - Batch Document Processing

**Purpose**: Process all supported files in a folder for parsing or structured data extraction.

**Important**: This tool calls the internal functions (`_parse_document_internal`, `_extract_data_internal`, `_create_parse_job_internal`) directly to avoid FunctionTool wrapper issues.

#### Input Validation
```python
# Validate operation mode
if operation not in ["parse", "extract"]:
    return error

# Extract mode requires schema
if operation == "extract" and not schema:
    return error("Schema required for extract")

# Check folder exists
folder = Path(folder_path)
if not folder.exists() or not folder.is_dir():
    return error
```

#### Core Logic Flow

1. **File Discovery**:
```python
SUPPORTED_EXTENSIONS = {
    # Images
    '.apng', '.bmp', '.dcx', '.dds', '.dib', '.gd', '.gif', 
    '.icns', '.jp2', '.jpeg', '.jpg', '.pcx', '.png', '.ppm', 
    '.psd', '.tga', '.tiff', '.tif', '.webp',
    # Documents
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.odp', '.odt'
}

# Filter by requested types
if file_types:  # e.g., "pdf,jpg"
    requested_exts = {f".{ext.lower()}" for ext in file_types.split(",")}
    allowed_exts = requested_exts & SUPPORTED_EXTENSIONS

# Find all matching files
for ext in allowed_exts:
    files.extend(folder.glob(f"*{ext}"))
    files.extend(folder.glob(f"*{ext.upper()}"))
```

2. **Size-Based Grouping**:
```python
small_files = []  # < 50MB - direct processing
large_files = []  # >= 50MB - use jobs

for file in all_files:
    size_mb = file.stat().st_size / (1024 * 1024)
    if size_mb < 50:
        small_files.append(file)
    else:
        large_files.append(file)
```

3. **Batch Processing Small Files**:
```python
# Process in batches to respect rate limits
for i in range(0, len(small_files), max_concurrent):
    batch = small_files[i:i + max_concurrent]
    
    # Create tasks based on operation - using internal functions!
    if operation == "parse":
        tasks = [_parse_document_internal(str(f)) for f in batch]
    else:  # extract
        # Custom async function for parse + extract
        async def parse_and_extract(fp=file_path, s=schema):
            parse_result = await _parse_document_internal(str(fp))
            extract_result = await _extract_data_internal(s, parse_result["markdown"])
            return combine_results(parse_result, extract_result)
        tasks.append(parse_and_extract())
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

4. **Large File Processing with Jobs**:
```python
# Create jobs for large files - using internal function!
jobs = []
for file in large_files:
    job_result = await _create_parse_job_internal(str(file))
    if job_result["status"] == "success":
        jobs.append({"file": file, "job_id": job_result["job_id"]})

# Monitor jobs
while pending_jobs:
    await asyncio.sleep(5)
    
    for job in pending_jobs[:]:
        status = await get_parse_job_status(job["job_id"])
        
        if status["status"] == "completed":
            pending_jobs.remove(job)
            # Process results based on operation
```

5. **Extract Mode Processing**:
```python
async def process_file_for_extraction(file_path, schema):
    # Step 1: Parse document
    parse_result = await parse_document(file_path)
    
    # Step 2: Extract structured data from markdown
    extract_result = await extract_data(
        schema=schema,
        markdown=parse_result["markdown"]
    )
    
    # Combine results
    return {
        "extraction": extract_result["extraction"],
        "markdown": parse_result["markdown"],
        "metadata": parse_result["metadata"]
    }
```

6. **Result Organization**:
```python
# Create output directory
output_dir = folder / "ade_results"
output_dir.mkdir(exist_ok=True)

# Save results for each file
for file in processed_files:
    file_dir = output_dir / f"{file.stem}_{operation}"
    file_dir.mkdir(exist_ok=True)
    
    if operation == "parse":
        # Save markdown, metadata, chunks
        (file_dir / "content.md").write_text(markdown)
        (file_dir / "metadata.json").write_text(json.dumps(metadata))
    else:  # extract
        # Save extracted data and source
        (file_dir / "data.json").write_text(json.dumps(extraction))
        (file_dir / "source.md").write_text(markdown)

# Save summary
(output_dir / "summary.json").write_text(json.dumps({
    "operation": operation,
    "total_files": len(all_files),
    "processed": len(processed_files),
    "failed": len(failed_files),
    "processing_time_s": processing_time
}))
```

#### Aggregation for Extract Mode
```python
aggregated_data = []

# Collect all extracted data
for result in extraction_results:
    aggregated_data.append({
        "source_file": file.name,
        **result["extraction"]  # Spread extracted fields
    })

# Save aggregated data
(output_dir / "extracted_data.json").write_text(
    json.dumps(aggregated_data, indent=2)
)
```

#### Design Decisions

1. **Why two operation modes?**
   - Parse: For general document processing, OCR, text extraction
   - Extract: For structured data extraction using schemas
   - Covers both exploratory and production use cases

2. **Why size-based grouping?**
   - Small files (<50MB): Fast enough for direct processing
   - Large files: Would timeout, need async jobs
   - Optimizes throughput vs resource usage

3. **Why max_concurrent parameter?**
   - Prevents API rate limit errors
   - Default of 15 balances speed and safety
   - User can adjust based on their API tier

4. **Why save to ade_results folder?**
   - Keeps original files untouched
   - Organized structure for downstream processing
   - Easy to find and delete results

5. **Why continue on errors?**
   - One corrupt file shouldn't stop batch processing
   - Failed files tracked in summary
   - Maximizes successful extractions

6. **Why aggregated_data for extract?**
   - Common need to combine extracted data
   - Ready for analysis or database import
   - Maintains source file tracking

#### Error Handling
```python
# Individual file errors don't stop batch
try:
    result = await parse_document(file)
except Exception as e:
    failed_files.append({
        "filename": file.name,
        "error": str(e)
    })
    continue  # Process next file

# Job failures tracked separately
if status["status"] == "failed":
    failed_files.append({
        "filename": job["file"].name,
        "error": status.get("failure_reason")
    })
```

#### Usage Patterns

**Bulk Invoice Processing**:
```python
# Extract data from hundreds of invoices
result = await process_folder(
    folder_path="/invoices/2024",
    operation="extract",
    schema=invoice_schema,
    file_types="pdf,jpg"  # PDFs and scanned images
)

# Access aggregated data
df = pd.DataFrame(result["aggregated_data"])
total_amount = df["total"].sum()
```

**Document Migration**:
```python
# Convert all documents to markdown
result = await process_folder(
    folder_path="/legacy_docs",
    operation="parse",
    model="dpt-2-latest"
)

# All markdown files now in ade_results/
```

**Mixed Document Analysis**:
```python
# Process everything, let API determine best approach
result = await process_folder(
    folder_path="/research_papers",
    save_results=True  # Keep all outputs
)
```

---

### 7. `health_check` - Server Status

**Purpose**: Verify server health and API connectivity.

#### Logic Flow
```python
result = {
    "status": "healthy",
    "server": "LandingAI ADE MCP Server",
    "api_key_configured": bool(API_KEY),
    "available_tools": [list of tools]
}

# Test API connectivity
if API_KEY:
    try:
        response = await client.get(
            f"{API_BASE_URL}/v1/ade/parse/jobs",
            params={'pageSize': 1}
        )
        if response.status_code == 200:
            result["api_connectivity"] = "connected"
        elif response.status_code == 401:
            result["api_connectivity"] = "invalid_api_key"
    except Exception as e:
        result["api_connectivity"] = f"connection_failed: {str(e)}"
```

#### Design Decisions
- **Why test with list jobs?** Lightweight endpoint for connectivity check
- **Why include tool list?** Helps users discover available functionality

---

## Common Patterns

### 1. Error Handling Pattern
```python
try:
    # Main logic
    response = await client.post(...)
    return handle_api_response(response)
except httpx.TimeoutException:
    return {"status": "error", "error": "Request timeout"}
except Exception as e:
    return {"status": "error", "error": str(e)}
```

### 2. Optional Parameter Pattern
```python
data = {}
if param1: data['param1'] = param1
if param2: data['param2'] = param2
# Only send data if not empty
response = await client.post(..., data=data if data else None)
```

### 3. File Size Management Pattern
```python
if len(content) > 500000:  # 500KB threshold
    # Save to file
    with open(output_file, 'w') as f:
        f.write(content)
    return {"data_file": output_file}
else:
    # Include in response
    return {"data": content}
```

### 4. Pre-signed URL Pattern
```python
# Create new client without auth headers
async with httpx.AsyncClient(timeout=60.0) as fetch_client:
    # No headers parameter!
    response = await fetch_client.get(pre_signed_url)
```

---

## Internal Functions Architecture

### Why Internal Functions?

The FastMCP framework replaces MCP-decorated functions with FunctionTool objects that cannot be called directly from within Python code. To solve this, the codebase uses a pattern where:

1. **Internal functions** (`_parse_document_internal`, `_extract_data_internal`, `_create_parse_job_internal`) contain all the actual API logic
2. **MCP tools** are thin wrappers that just call the internal functions
3. **process_folder** calls the internal functions directly, avoiding FunctionTool issues

### Internal Function List

- `_parse_document_internal()` - Document parsing logic
- `_extract_data_internal()` - Data extraction logic  
- `_create_parse_job_internal()` - Job creation logic

### Benefits

- **No code duplication** - Logic exists in one place
- **Testability** - Internal functions can be tested independently
- **Flexibility** - Any function can call internal functions
- **Clean separation** - API logic separated from MCP interface

## Key Insights

1. **Context Window Management**: The server automatically detects large responses and saves them to files, preventing Claude's context window from being overwhelmed.

2. **Progressive Disclosure**: Tools provide summaries and previews before full data, letting users decide what to access.

3. **Fail-Safe Design**: When auto-fetch fails, the tool still returns the URL so users can access results manually.

4. **Transparency**: URLs and file paths are printed for visibility, making the process debuggable.

5. **Smart Detection**: The server intelligently detects whether inputs are file paths or content strings, improving usability.

6. **S3 Authentication**: Understanding that pre-signed URLs include auth in the URL itself is crucial for successful downloads.

7. **Internal Functions Pattern**: Solves the FunctionTool wrapper issue by separating API logic from MCP decorators.

This architecture ensures reliable document processing while managing memory efficiently and providing clear feedback throughout the process.