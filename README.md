# LandingAI ADE MCP Server

A Model Context Protocol (MCP) server providing direct integration with LandingAI's Agentic Document Extraction (ADE) API. Extract text, tables, and structured data from PDFs, images, and office documents.

## Features

- 📄 **Document Parsing** - Parse entire documents and return markdown output
- 🔍 **Data Extraction** - Extract structured data using JSON schemas
- ⚡ **Parse Jobs** - Handle large documents with background processing
- 🛡️ **Zero Data Retention** - Privacy-focused processing support

## Installation

### Prerequisites

- Python 3.9 or higher
- LandingAI API key from [LandingAI](https://docs.landing.ai/ade/agentic-api-key)

### Option 1: Using uv (Recommended - Simplest)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that handles virtual environments automatically.

#### Install uv (if not already installed)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

#### Set up the project
```bash
# Clone the repository
git clone https://github.com/avaxia8/landingai-ade-mcp.git
cd landingai-ade-mcp

# Install dependencies with uv
uv sync

# Or if starting fresh:
uv init
uv add fastmcp httpx pydantic python-multipart aiofiles
```

### Option 2: Using pip with Virtual Environment

```bash
# Clone the repository
git clone https://github.com/avaxia8/landingai-ade-mcp.git
cd landingai-ade-mcp

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```


## Configuration

### Set Your API Key

Get your API key from [LandingAI](https://docs.landing.ai/ade/agentic-api-key)

```bash
export LANDINGAI_API_KEY="your-api-key-here"
```

### Claude Desktop Configuration

#### Configuration File Location

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`  
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

#### Configuration Examples

##### Using uv (Recommended)

```json
{
  "mcpServers": {
    "landingai-ade-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/landingai-ade-mcp",
        "run",
        "python",
        "-m",
        "server"
      ],
      "env": {
        "LANDINGAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

##### Using Virtual Environment

```json
{
  "mcpServers": {
    "landingai-ade-mcp": {
      "command": "/path/to/landingai-ade-mcp/venv/bin/python",
      "args": [
        "/path/to/landingai-ade-mcp/server.py"
      ],
      "env": {
        "LANDINGAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```


### After Configuration

1. Save the configuration file
2. **Restart Claude Desktop completely** (quit and reopen)
3. The server should appear as "landingai-ade-mcp" in your MCP servers


## Available Tools



### `parse_document`
Parse documents to extract content with metadata.

```python
# Parse a local file
result = await parse_document(
    document_path="/path/to/document.pdf",
    model="dpt-2-latest",  # optional
    split="page"  # optional, for page-level splits
)

# Parse from URL
result = await parse_document(
    document_url="https://example.com/document.pdf"
)
```

### `extract_data`
Extract structured data from markdown using a JSON schema.

```python
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"}
    }
}

# Extract from markdown content string
result = await extract_data(
    schema=schema,
    markdown="# Invoice\nInvoice #123\nTotal: $100.00"
)

# Or extract from a markdown file
result = await extract_data(
    schema=schema,
    markdown="/path/to/document.md"  # Will detect if it's a file path
)

# Or extract from URL
result = await extract_data(
    schema=schema,
    markdown_url="https://example.com/document.md"
)
```

### `create_parse_job`
Create a parse job for large documents (>50MB recommended).

```python
job = await create_parse_job(
    document_path="/path/to/large_document.pdf",
    split="page"  # optional
)
job_id = job["job_id"]
```

### `get_parse_job_status`
Check status and retrieve results of a parse job.

```python
status = await get_parse_job_status(job_id)

# Check status
if status["status"] == "completed":
    # For small files, data is included directly
    # For large files (>1MB), data is auto-fetched from output_url
    results = status["data"]
elif status["status"] == "processing":
    print(f"Progress: {status['progress'] * 100:.1f}%")
```

### `download_from_url`
Download results from a pre-signed URL (optional - only if auto-fetch fails).

```python
# Only needed if get_parse_job_status fails to auto-fetch
results = await download_from_url(output_url)
```

### `list_parse_jobs`
List all parse jobs with filtering and pagination.

```python
jobs = await list_parse_jobs(
    page=0,  # optional, default 0
    pageSize=10,  # optional, 1-100, default 10
    status="completed"  # optional filter
)
```
### `process_folder`
Process all supported files in a folder - parse documents or extract structured data.

**Supported formats:** 
- Images: APNG, BMP, DCX, DDS, DIB, GD, GIF, ICNS, JP2, JPEG, JPG, PCX, PNG, PPM, PSD, TGA, TIFF, WEBP
- Documents: PDF, DOC, DOCX, PPT, PPTX, ODP, ODT

```python
# Parse all PDFs in a folder
result = await process_folder(
    folder_path="/path/to/documents",
    operation="parse",  # or "extract" for structured data
    file_types="pdf",   # optional filter
    model="dpt-2-latest"
)

# Extract structured data from all documents
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "date": {"type": "string"}
    }
}

result = await process_folder(
    folder_path="/path/to/invoices",
    operation="extract",
    schema=schema,
    file_types="pdf,jpg"  # Process PDFs and images
)

# Process everything with defaults
result = await process_folder(
    folder_path="/path/to/mixed_documents"
)
```

**Features:**
- Automatic file size detection (uses direct parsing for <10MB, jobs for larger)
- Concurrent processing with rate limiting
- Progress tracking for long-running operations
- Organized output in `ade_results` folder
- Aggregated data for extraction operations
- Continues processing even if individual files fail

### `health_check`
Check server status and API connectivity.

```python
health = await health_check()
# Returns server status, API connectivity, available tools
```

## Usage Examples

### Basic Document Processing

```python
# 1. Parse a document
parse_result = await parse_document(
    document_path="/path/to/invoice.pdf"
)

# 2. Extract structured data
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {
            "type": "string",
            "description": "Invoice number"
        },
        "date": {"type": "string"},
        "total": {"type": "number"}
    }
}

extract_result = await extract_data(
    schema=schema,
    markdown=parse_result["markdown"]
)

print(extract_result["extraction"])
```

### Batch Processing Folders

```python
# Process an entire folder of invoices
result = await process_folder(
    folder_path="/path/to/invoices",
    operation="extract",
    schema={
        "type": "object",
        "properties": {
            "invoice_no": {"type": "string"},
            "vendor": {"type": "string"},
            "total": {"type": "number"},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "amount": {"type": "number"}
                    }
                }
            }
        }
    },
    file_types="pdf,jpg,png"  # Process scanned and digital invoices
)

# Access results
print(f"Processed: {result['summary']['processed']} files")
print(f"Failed: {result['summary']['failed']} files")
print(f"Results saved to: {result['results_path']}")

# Aggregated data for all invoices
for invoice in result.get('aggregated_data', []):
    print(f"Invoice {invoice['invoice_no']}: ${invoice['total']}")
```

**Output Structure:**
```
/your_folder/
├── documents/              # Original files
└── ade_results/           # Processing results
    ├── summary.json       # Overall summary
    ├── extracted_data.json # All extracted data (for extract operation)
    ├── invoice1_extract/
    │   ├── data.json      # Extracted structured data
    │   └── source.md      # Original markdown
    └── invoice2_extract/
        └── ...
```

### Processing Large Files

```python
# 1. Create parse job
job = await create_parse_job(
    document_path="/path/to/large_document.pdf"
)

# 2. Monitor progress
import asyncio
while True:
    status = await get_parse_job_status(job["job_id"])
    
    if status["status"] == "completed":
        # Get results
        if "data" in status:
            results = status["data"]
        else:
            # Download from URL if >1MB
            results_url = status["output_url"]
        break
    
    print(f"Progress: {status['progress'] * 100:.1f}%")
    await asyncio.sleep(5)
```

### Zero Data Retention

```python
# Process with ZDR - results saved to your URL
job = await create_parse_job(
    document_path="/path/to/sensitive.pdf",
    output_save_url="https://your-storage.com/results/"
)

status = await get_parse_job_status(job["job_id"])
# Results at status["output_url"], not in response
```

## File Size Guidelines

- **< 10MB**: Use `parse_document` directly
- **10-50MB**: Consider parse jobs for better performance
- **> 50MB**: Always use `create_parse_job`

## Error Handling

```python
result = await parse_document(document_path="/path/to/file.pdf")

if result.get("status") == "error":
    print(f"Error: {result['error']}")
    print(f"Status Code: {result.get('status_code')}")
else:
    # Process successful result
    markdown = result["markdown"]
```

### Common Error Codes

- `401`: Invalid API key
- `413`: File too large (use parse jobs)
- `422`: Validation error
- `429`: Rate limit exceeded

## Troubleshooting

### Common Issues and Solutions

#### "Could not connect to MCP server"

1. **Python not found**: Make sure the Python path in your config is correct
   ```bash
   # Find your Python path
   which python3
   ```

2. **Module not found errors**: Dependencies aren't installed in the Python environment
   - If using uv: Run `uv sync` in the project directory
   - If using venv: Activate it and run `pip install -r requirements.txt`
   - Check that the Python path in config matches your environment

3. **spawn python ENOENT**: The system can't find Python
   - Use the full path to Python (e.g., `/usr/bin/python3` instead of just `python`)
   - For virtual environments, use the full path to the venv's Python

#### "Server disconnected"

1. **Check the server can run manually**:
   ```bash
   cd /path/to/landingai-ade-mcp
   python server.py
   # Should see: "Starting LandingAI ADE MCP Server"
   ```

2. **Check API key is set**:
   ```bash
   echo $LANDINGAI_API_KEY
   ```

3. **Check dependencies are installed**:
   ```bash
   python -c "import fastmcp, httpx, pydantic"
   # Should complete without errors
   ```

#### "ModuleNotFoundError: No module named 'fastmcp'"

This means fastmcp isn't installed in the Python environment being used:

- **If using virtual environment**: The config is pointing to the wrong Python
- **Solution**: Use uv or ensure the Python path matches your environment

#### Platform-Specific Issues

**macOS**: If you installed Python with Homebrew, the path might be `/opt/homebrew/bin/python3` (Apple Silicon) or `/usr/local/bin/python3` (Intel)

**Windows**: Use forward slashes in paths or escape backslashes: `C:/path/to/python.exe` or `C:\\path\\to\\python.exe`

**Linux**: Some systems use `python3` instead of `python`. Always use `python3` for clarity.

### Debug Steps

1. **Test the server standalone**:
   ```bash
   python server.py
   ```

2. **Check MCP communication**:
   ```bash
   echo '{"jsonrpc": "2.0", "method": "initialize", "id": 1}' | python server.py
   ```

3. **Verify configuration**:
   - Open Claude Desktop developer settings
   - Check the logs for specific error messages
   - Ensure all paths are absolute, not relative

4. **Validate API key**:
   ```bash
   python -c "import os; print('API Key set:', bool(os.environ.get('LANDINGAI_API_KEY')))"
   ```

## Requirements

- Python 3.8+
- API key from [LandingAI](https://landing.ai)

## Dependencies

- `fastmcp>=0.1.0` - MCP server framework
- `httpx>=0.24.0` - HTTP client
- `pydantic>=2.0.0` - Data validation
- `python-multipart>=0.0.6` - Form handling
- `aiofiles>=23.0.0` - Async file operations

## Why Local Deployment?

This server runs locally to ensure:
- **Privacy**: Your API keys and documents stay on your machine
- **Security**: No third-party access to your data
- **Control**: Direct management of API usage and costs

## API Documentation

- [LandingAI API Reference](https://docs.landing.ai/api-reference)
- [Supported File Types](https://docs.landing.ai/ade/ade-file-types)
