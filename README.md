A Model Context Protocol server that provides unstructured document processing capabilities. 
This server enables LLMs to extract and use content from an unstructured document.

**This repo is work in progress, proceed with caution :)**

Supported file types:

```
{".abw", ".bmp", ".csv", ".cwk", ".dbf", ".dif", ".doc", ".docm", ".docx", ".dot",
 ".dotm", ".eml", ".epub", ".et", ".eth", ".fods", ".gif", ".heic", ".htm", ".html",
 ".hwp", ".jpeg", ".jpg", ".md", ".mcw", ".mw", ".odt", ".org", ".p7s", ".pages",
 ".pbd", ".pdf", ".png", ".pot", ".potm", ".ppt", ".pptm", ".pptx", ".prn", ".rst",
 ".rtf", ".sdp", ".sgl", ".svg", ".sxg", ".tiff", ".txt", ".tsv", ".uof", ".uos1",
 ".uos2", ".web", ".webp", ".wk2", ".xls", ".xlsb", ".xlsm", ".xlsx", ".xlw", ".xml",
 ".zabw"}
```

Prerequisites: 
You'll need:
* Unstructured API key. [Learn how to obtain one here](https://docs.unstructured.io/api-reference/partition/overview#get-started)
* Claude Desktop installed locally

Quick TLDR on how to add this MCP to your Claude Desktop:
1. Clone the repo and set up the UV environment.
2. Create a `.env` file in the root directory and add the following env variable: `UNSTRUCTURED_API_KEY`.
3. Run the MCP server: `uv run doc_processor.py`
4. Go to `~/Library/Application Support/Claude/` and create a `claude_desktop_config.json`. In that file add:
```
{
    "mcpServers": {
        "unstructured_doc_processor": {
            "command": "PATH/TO/YOUR/UV",
            "args": [
                "--directory",
                "ABSOLUTE/PATH/TO/YOUR/unstructured-mcp/",
                "run",
                "doc_processor.py"
            ],
            "disabled": false
        }
    }
}
```
5. Restart Claude Desktop. You should now be able to use the MCP.
