import os
from dotenv import load_dotenv
import boto3
import time
import json
from botocore.exceptions import ClientError
import logging
from unstructured_client import UnstructuredClient
from unstructured_client.models.shared import (
    S3SourceConnectorConfigInput,
    CreateSourceConnector,
    SourceConnectorType)
from unstructured_client.models.operations import (CreateSourceRequest,
                                                   CreateDestinationRequest,
                                                   DeleteWorkflowRequest,
                                                   DeleteSourceRequest,
                                                   DeleteDestinationRequest)
from unstructured_client.models.shared import (
    S3DestinationConnectorConfigInput,
    CreateDestinationConnector,
    DestinationConnectorType)
from unstructured_client.models.shared import (
    WorkflowNode,
    WorkflowNodeType,
    WorkflowType
)
from datetime import datetime
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("unstructured_doc_processor")


def load_environment_variables() -> None:
    """
    Load environment variables from .env file.
    Raises an error if critical environment variables are missing.
    """
    load_dotenv()
    required_vars = [
        "AWS_S3_SOURCE_BUCKET",
        "AWS_S3_DESTINATION_BUCKET",
        "AWS_KEY",
        "AWS_SECRET",
        "UNSTRUCTURED_API_KEY"
    ]

    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Missing required environment variable: {var}")


def upload_to_s3(file_path, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-2", object_name=None):
    """
    Upload a file to an S3 bucket

    Parameters:
    file_path (str): Path to the file to upload
    bucket_name (str): Name of the bucket to upload to
    object_name (str): S3 object name. If not specified, file_name from file_path will be used
    """
    # If S3 object_name was not specified, use file_name from file_path
    if object_name is None:
        object_name = os.path.basename(file_path)

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )

    try:
        s3_client.upload_file(file_path, bucket_name, object_name)
        print(f"Successfully uploaded {file_path} to {bucket_name}/{object_name}")
    except ClientError as e:
        logging.error(e)
        print(f"Error uploading file: {e}")


def download_s3_file(bucket_name, file_name, local_dir, aws_access_key, aws_secret_key, region_name="us-east-2"):
    """
    Downloads a specific file from an S3 bucket to a local directory.

    :param bucket_name: Name of the S3 bucket.
    :param file_name: Name of the original file to download JSON for from S3.
    :param local_dir: Local directory where the file should be saved.
    :param aws_access_key: (Optional) AWS Access Key ID.
    :param aws_secret_key: (Optional) AWS Secret Access Key.
    :param region_name: AWS region where the bucket is located.
    :return: Path to the downloaded file if successful, None otherwise
    """

    file_key = f"{file_name}.json"
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
    else:
        s3_client = boto3.client("s3", region_name=region_name)

    try:
        # Ensure the local directory exists
        os.makedirs(local_dir, exist_ok=True)

        # Set the local file path
        local_file_path = os.path.join(local_dir, os.path.basename(file_key))

        # Create subdirectories if necessary
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        print(f"Downloading {file_key} from bucket {bucket_name} to {local_file_path}...")
        s3_client.download_file(bucket_name, file_key, local_file_path)
        print("Download complete.")

        return local_file_path

    except Exception as e:
        print(f"Error downloading file {file_key}: {str(e)}")
        return None


def get_unstructured_client(unstructured_api_key):
    return UnstructuredClient(api_key_auth=unstructured_api_key)


def create_s3_source_connector(unstructured_client, s3_bucket_name, aws_key, aws_secret):
    """ Creates an S3 source connector in Unstructured platform and returns a connector's id
    """
    source_connector_config = S3SourceConnectorConfigInput(
        remote_url=f"s3://{s3_bucket_name}",
        key=aws_key,
        secret=aws_secret,
    )
    unique_source_connector_suffix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    response = unstructured_client.sources.create_source(
        request=CreateSourceRequest(
            create_source_connector=CreateSourceConnector(
                name=f"s3-source-{unique_source_connector_suffix}",
                type=SourceConnectorType.S3,
                config=source_connector_config,
            )
        )
    )
    return response.source_connector_information.id


def create_s3_destination_connector(unstructured_client, s3_bucket_name, aws_key, aws_secret):
    """ Creates an S3 source connector in Unstructured platform and returns a connector's id
    """
    destination_connector_config = S3DestinationConnectorConfigInput(
        remote_url=f"s3://{s3_bucket_name}",
        key=aws_key,
        secret=aws_secret,
    )
    unique_destination_connector_suffix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    response = unstructured_client.destinations.create_destination(
        request=CreateDestinationRequest(
            create_destination_connector=CreateDestinationConnector(
                name=f"s3-destination-{unique_destination_connector_suffix}",
                type=DestinationConnectorType.S3,
                config=destination_connector_config,
            )
        )
    )

    return response.destination_connector_information.id


def create_auto_workflow(unstructured_client, source_connector_id, destination_connector_id):
    """ Creates a workflow from a source to destination in Unstructured platform, returns workflow id
    """

    # Partition the content by using a vision language model (VLM).
    parition_node = WorkflowNode(
        name="Partitioner",
        subtype="vlm",
        type=WorkflowNodeType.PARTITION,
        settings={
            "provider": "anthropic",
            "provider_api_key": None,
            "model": "claude-3-5-sonnet-20241022",
            "output_format": "text/html",
            "user_prompt": None,
            "format_html": True,
            "unique_element_ids": True,
            "is_dynamic": True,
            "allow_fast": True
        }
    )
    # Summarize each detected image.
    image_summarizer_node = WorkflowNode(
        name="Image summarizer",
        subtype="openai_image_description",
        type=WorkflowNodeType.PROMPTER,
        settings={}
    )
    # Summarize each detected table.
    table_summarizer_node = WorkflowNode(
        name="Table summarizer",
        subtype="anthropic_table_description",
        type=WorkflowNodeType.PROMPTER,
        settings={}
    )
    unique_workflow_suffix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    response = unstructured_client.workflows.create_workflow(
        request={
            "create_workflow": {
                "name": f"s3-to-s3-custom-workflow-{unique_workflow_suffix}",
                "source_id": source_connector_id,
                "destination_id": destination_connector_id,
                "workflow_type": WorkflowType.CUSTOM,
                "workflow_nodes": [parition_node, image_summarizer_node, table_summarizer_node]
            }
        }
    )

    return response.workflow_information.id


def run_workflow(unstructured_client, workflow_id):
    res = unstructured_client.workflows.run_workflow(
            request={
                "workflow_id": workflow_id,
            }
        )
    print(res.job_information)


def get_latest_job_id(unstructured_client, workflow_id):
    response = unstructured_client.jobs.list_jobs(
        request={
            "workflow_id": workflow_id
        }
    )
    last_job = response.response_list_jobs[0]
    return last_job.id


def poll_job_status(unstructured_client, job_id):
    while True:
        response = unstructured_client.jobs.get_job(
            request={
                "job_id": job_id
            }
        )

        job = response.job_information
        if job.status == "SCHEDULED":
            print("Job is scheduled, polling again in 10 seconds...")
            time.sleep(10)
        elif job.status == "IN_PROGRESS":
            print("Job is in progress, polling again in 10 seconds...")
            time.sleep(10)
        else:
            print("Job is completed")
            break

    return job


def delete_workflow(unstructured_client, workflow_id):
    response = unstructured_client.workflows.delete_workflow(
        request=DeleteWorkflowRequest(
            workflow_id=workflow_id
        ))
    print(response.raw_response)


def delete_source_connector(unstructured_client, source_connector_id):
    response = unstructured_client.sources.delete_source(
        request=DeleteSourceRequest(
            source_id=source_connector_id
        ))

    print(response.raw_response)

def delete_destination_connector(unstructured_client, destination_connector_id):
    response = unstructured_client.destinations.delete_destination(
        request=DeleteDestinationRequest(
            destination_id=destination_connector_id
        )
    )
    print(response.raw_response)


def empty_s3_bucket(bucket_name, aws_access_key, aws_secret_key, region_name="us-east-2"):
    """
    Deletes all files from an S3 bucket and returns a list of deleted files.
    """
    if aws_access_key and aws_secret_key:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
    else:
        s3_client = boto3.client("s3", region_name=region_name)

    deleted_files = []

    # List objects in the bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)

    if "Contents" not in response:
        print(f"No files found in bucket: {bucket_name}")
        return deleted_files

    # Delete each file
    for obj in response["Contents"]:
        file_key = obj["Key"]
        print(f"Deleting {file_key}...")
        s3_client.delete_object(Bucket=bucket_name, Key=file_key)
        deleted_files.append(file_key)

    # Handle pagination if there are more than 1000 objects
    while response.get('IsTruncated', False):
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            ContinuationToken=response['NextContinuationToken']
        )

        if "Contents" in response:
            for obj in response["Contents"]:
                file_key = obj["Key"]
                print(f"Deleting {file_key}...")
                s3_client.delete_object(Bucket=bucket_name, Key=file_key)
                deleted_files.append(file_key)

    return deleted_files


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
async def get_processed_doc(filepath: str) -> str:
    """Get the text from the given document.
    Args:
        filepath: local file path to the document
    """

    # Check is file exists in given path
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
        return "File extension not supported by Unstructured Platform"

    upload_to_s3(filepath,
                 os.getenv("AWS_S3_SOURCE_BUCKET"),
                 os.getenv("AWS_KEY"),
                 os.getenv("AWS_SECRET"))
    unstructured_client = get_unstructured_client(os.getenv("UNSTRUCTURED_API_KEY"))
    source_connector_id = create_s3_source_connector(unstructured_client,
                               os.getenv("AWS_S3_SOURCE_BUCKET"),
                               os.getenv("AWS_KEY"),
                               os.getenv("AWS_SECRET"))

    destination_connector_id = create_s3_destination_connector(unstructured_client,
                                                               os.getenv("AWS_S3_DESTINATION_BUCKET"),
                                                               os.getenv("AWS_KEY"),
                                                               os.getenv("AWS_SECRET"))

    workflow_id = create_auto_workflow(unstructured_client, source_connector_id, destination_connector_id)
    run_workflow(unstructured_client, workflow_id)
    job_id = get_latest_job_id(unstructured_client, workflow_id)
    job = poll_job_status(unstructured_client, job_id)
    # At this point the job is complete

    # TODO: add a check to see if the job is successful
    print(f"Unstructured Platform Completed Processing Job: {job_id}")

    # download the file from the destination bucket
    local_dir = "processed_files"
    file_basename = os.path.basename(filepath)
    download_s3_file(os.getenv("AWS_S3_DESTINATION_BUCKET"),
                     file_basename,
                     local_dir,
                     os.getenv("AWS_KEY"),
                     os.getenv("AWS_SECRET"))

    # Cleanup:
    delete_workflow(unstructured_client, workflow_id)
    delete_source_connector(unstructured_client, source_connector_id)
    delete_destination_connector(unstructured_client, destination_connector_id)

    deleted_source_files = empty_s3_bucket(os.getenv("AWS_S3_SOURCE_BUCKET"), os.getenv("AWS_KEY"),
                               os.getenv("AWS_SECRET"))
    print("Files deleted from the source S3 bucket: ", deleted_source_files)

    deleted_processed_files = empty_s3_bucket(os.getenv("AWS_S3_DESTINATION_BUCKET"), os.getenv("AWS_KEY"),
                               os.getenv("AWS_SECRET"))
    print("Files deleted from the output S3 bucket: ", deleted_processed_files)

    output_json_file_path = os.path.join(local_dir, f"{file_basename}.json")

    document_text = json_to_text(output_json_file_path)

    return document_text


if __name__ == "__main__":
    load_environment_variables()
    # Initialize and run the server
    mcp.run(transport='stdio')