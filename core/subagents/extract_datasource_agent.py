import json
import yaml
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field
import docx  # python-docx for reading Word documents
import mammoth  # Alternative for Word document processing


# --- Data Models ---
class ConnectionParams(BaseModel):
    """Connection parameters for different data source types"""
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    account: Optional[str] = None
    warehouse: Optional[str] = None
    s3_bucket: Optional[str] = None
    aws_access_key: Optional[str] = None
    aws_secret_key: Optional[str] = None
    # Add other connection parameters as needed


class DataSource(BaseModel):
    """Data source definition"""
    name: str = Field(description="Name of the data source")
    type: str = Field(description="Type of data source (postgresql, snowflake, parquet, etc.)")
    connection_params: ConnectionParams = Field(description="Connection parameters")
    tables_or_files: List[str] = Field(description="List of tables or files")
    description: str = Field(description="Description of the data source")


class RequirementsOutput(BaseModel):
    """Output schema for the YAML requirements"""
    data_sources: List[DataSource] = Field(description="List of data sources")


class DocumentInput(BaseModel):
    """Input schema for document processing"""
    document_content: str = Field(description="Content of the requirements document")


# --- Tools ---
def extract_document_content(file_path: str) -> str:
    """
    Extract text content from Word document or text file.
    Supports .docx, .doc, and .txt files.
    """
    try:
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.docx':
            # Use python-docx for .docx files
            doc = docx.Document(file_path)
            content = []
            for paragraph in doc.paragraphs:
                content.append(paragraph.text)
            return '\n'.join(content)

        elif file_path.suffix.lower() == '.doc':
            # Use mammoth for .doc files (converts to HTML then extract text)
            with open(file_path, "rb") as doc_file:
                result = mammoth.extract_raw_text(doc_file)
                return result.value

        elif file_path.suffix.lower() == '.txt':
            # Plain text files
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        else:
            return f"Unsupported file format: {file_path.suffix}"

    except Exception as e:
        return f"Error reading document: {str(e)}"


def save_yaml_output(yaml_content: str, output_path: str) -> str:
    """Save the generated YAML to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        return f"YAML successfully saved to: {output_path}"
    except Exception as e:
        return f"Error saving YAML: {str(e)}"


def validate_yaml_structure(yaml_content: str) -> str:
    """Validate the generated YAML structure"""
    try:
        parsed_yaml = yaml.safe_load(yaml_content)

        # Check if data_sources key exists
        if 'data_sources' not in parsed_yaml:
            return "Error: 'data_sources' key is missing from YAML"

        # Validate each data source
        data_sources = parsed_yaml['data_sources']
        if not isinstance(data_sources, list):
            return "Error: 'data_sources' should be a list"

        required_fields = ['name', 'type', 'connection_params', 'tables_or_files', 'description']

        for i, source in enumerate(data_sources):
            for field in required_fields:
                if field not in source:
                    return f"Error: Data source {i + 1} is missing required field '{field}'"

        return "YAML structure is valid"

    except yaml.YAMLError as e:
        return f"Error: Invalid YAML format - {str(e)}"
    except Exception as e:
        return f"Error validating YAML: {str(e)}"


# --- Agent Configuration ---
def create_extract_source_datasource_agent() -> LlmAgent:
    """Create the data engineering requirements to YAML conversion agent"""

    agent = LlmAgent(
        model="gemini-2.0-flash",
        name="rqrmnts_yaml_converter",
        description="Converts data engineering requirement documents to structured YAML configuration",
        instruction="""You are a specialized Data Engineering Agent that converts detailed requirement documents into structured YAML configurations for data sources.

**Your Task:**
Convert a data engineering requirements document into a specific YAML structure that defines data sources with their connection parameters.

**Input Document Sections to Analyze:**
1. **Opportunity Description** - Understand the overall project context
2. **Business Impact** - Understand the business value and scope
3. **Data Sources** - CRITICAL: Extract all mentioned data sources, databases, files, APIs
4. **Data Targets** - Understand where data needs to go
5. **Success Metrics** - Understand measurement requirements

**YAML Structure to Generate:**
```yaml
data_sources:
  - name: "Descriptive Name"
    type: "database_type"  # postgresql, snowflake, parquet, mysql, etc.
    connection_params:
      # Database specific parameters
      host: "hostname"
      port: 5432
      database: "db_name" 
      user: "username"
      password: "password"
      # OR Cloud specific parameters
      account: "account.snowflakecomputing.com"
      warehouse: "WAREHOUSE_NAME"
      # OR File specific parameters  
      s3_bucket: "bucket-name"
      aws_access_key: "key"
      aws_secret_key: "secret"
    tables_or_files:
      - "table_name_1"
      - "table_name_2" 
      - "s3://bucket/path/file.parquet"
    description: "Clear description of what this source contains"
```

**Critical Instructions:**
1. **Extract ALL data sources** mentioned in the document - databases, files, APIs, data lakes, warehouses
2. **Infer connection types** from context clues (PostgreSQL, MySQL, Snowflake, S3, etc.)
3. **Generate realistic connection parameters** based on typical enterprise patterns:
   - Use company domain patterns for hostnames (e.g., "claims-db.company.com")  
   - Use standard ports (5432 for PostgreSQL, 3306 for MySQL, etc.)
   - Create descriptive usernames (e.g., "analytics_user", "discovery_user")
   - Use placeholder passwords that indicate security ("secure_password", "vault_managed")
4. **Extract table/file names** mentioned or infer logical names from context
5. **Create meaningful descriptions** that explain the business purpose of each source

**When information is missing:**
- Make reasonable inferences based on industry standards
- Use placeholder values that clearly indicate they need to be filled in
- Prioritize creating a complete, valid YAML structure

**Output Format:**
Respond with ONLY the YAML content, properly formatted and indented. Do not include markdown code blocks or additional explanations unless there are critical issues.""",

        tools=[extract_document_content, save_yaml_output, validate_yaml_structure],
        input_schema=DocumentInput,
        output_key="generated_yaml"
    )

    return agent
