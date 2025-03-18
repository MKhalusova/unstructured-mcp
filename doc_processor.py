import os
from dotenv import load_dotenv
import json
from unstructured_client import UnstructuredClient
from typing import AsyncIterator
from dataclasses import dataclass
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP, Context
from unstructured_client.models import operations, shared


@dataclass
class AppContext:
    client: UnstructuredClient


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage Unstructured API client lifecycle"""
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        raise ValueError("UNSTRUCTURED_API_KEY environment variable is required")

    client = UnstructuredClient(api_key_auth=api_key)
    try:
        yield AppContext(client=client)
    finally:
        # No cleanup needed for the API client
        pass


# Create MCP server instance
mcp = FastMCP("unstructured-mcp", lifespan=app_lifespan, dependencies=["unstructured-client", "python-dotenv"])
# Local directory to store processed files
PROCESSED_FILES_FOLDER = "processed_files"


def load_environment_variables() -> None:
    """
    Load environment variables from .env file.
    Raises an error if critical environment variables are missing.
    """
    load_dotenv()
    required_vars = [
        "UNSTRUCTURED_API_KEY"
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")


def json_to_text(file_path) -> str:
    with open(file_path, 'r') as file:
        elements = json.load(file)

    doc_texts = []

    for element in elements:
        text = element.get("text", "").strip()
        element_type = element.get("type", "")
        metadata = element.get("metadata", {})

        if element_type == "Title":
            doc_texts.append(f"<h1> {text}</h1><br>")
        elif element_type == "Header":
            doc_texts.append(f"<h2> {text}</h2><br/>")
        elif element_type == "NarrativeText" or element_type == "UncategorizedText":
            doc_texts.append(f"<p>{text}</p>")
        elif element_type == "ListItem":
            doc_texts.append(f"<li>{text}</li>")
        elif element_type == "PageNumber":
            doc_texts.append(f"Page number: {text}")
        elif element_type == "Table":
            table_html = metadata.get("text_as_html", "")
            doc_texts.append(table_html)  # Keep the table as HTML
        else:
            doc_texts.append(text)

    return " ".join(doc_texts)


@mcp.tool()
async def process_document(ctx: Context, filepath: str) -> str:
    """
    Sends document to process with Unstructured, return the content of the document
     Args:
    filepath: path to the document
    """

    if not os.path.isfile(filepath):
        return "File does not exist"

    # Check is file extension is supported
    _, ext = os.path.splitext(filepath)
    supported_extensions = {".abw", ".bmp", ".csv", ".cwk", ".dbf", ".dif", ".doc", ".docm", ".docx", ".dot",
                            ".dotm", ".eml", ".epub", ".et", ".eth", ".fods", ".gif", ".heic", ".htm", ".html",
                            ".hwp", ".jpeg", ".jpg", ".md", ".mcw", ".mw", ".odt", ".org", ".p7s", ".pages",
                            ".pbd", ".pdf", ".png", ".pot", ".potm", ".ppt", ".pptm", ".pptx", ".prn", ".rst",
                            ".rtf", ".sdp", ".sgl", ".svg", ".sxg", ".tiff", ".txt", ".tsv", ".uof", ".uos1",
                            ".uos2", ".web", ".webp", ".wk2", ".xls", ".xlsb", ".xlsm", ".xlsx", ".xlw", ".xml",
                            ".zabw"}

    if ext.lower() not in supported_extensions:
        return "File extension not supported by Unstructured"

    client = ctx.request_context.lifespan_context.client
    file_basename = os.path.basename(filepath)

    req = operations.PartitionRequest(
        partition_parameters=shared.PartitionParameters(
            files=shared.Files(
                content=open(filepath, "rb"),
                file_name=filepath,
            ),
            strategy=shared.Strategy.AUTO,
        ),
    )

    try:
        res = client.general.partition(request=req)
        element_dicts = [element for element in res.elements]
        json_elements = json.dumps(element_dicts, indent=2)
        output_json_file_path = os.path.join(PROCESSED_FILES_FOLDER, f"{file_basename}.json")
        with open(output_json_file_path, "w") as file:
            file.write(json_elements)

        return json_to_text(output_json_file_path)
    except Exception as e:
        return f"The following exception happened during file processing: {e}"


if __name__ == "__main__":
    load_environment_variables()
    # Initialize and run the server
    mcp.run(transport='stdio')