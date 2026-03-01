import os
import logging
import re
import json
from typing import Dict, Any

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.graph.state import VideoAuditState
from backend.src.services.video_indexer import VideoIndexerService

logger = logging.getLogger("brand-guardian")
logging.basicConfig(level=logging.INFO)


def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    video_url = state.get("video_url")
    video_id = state.get("video_id", "vid_demo")

    if not video_url:
        return {
            "error": ["Missing video_url in state"],
            "final_status": "FAIL",
        }

    logger.info(f"---- [Node:Indexer] Processing: {video_url}")
    local_filename = "temp_audit_video.mp4"

    try:
        vi_service = VideoIndexerService()

        # Validate YouTube URL
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(
                video_url,
                output_path=local_filename
            )
        else:
            raise Exception("Please provide a valid YouTube URL.")

        # Upload to Azure Video Indexer
        azure_video_id = vi_service.upload_video(
            local_path,
            video_name=video_id
        )

        logger.info(f"Upload success. Azure ID: {azure_video_id}")

        # Cleanup local file
        if os.path.exists(local_path):
            os.remove(local_path)

        # Wait for processing
        raw_insights = vi_service.wait_for_processing(azure_video_id)

        # Extract structured data
        clean_data = vi_service.extract_data(raw_insights)

        logger.info("---- [Node:Indexer] Extraction Complete ----")

        return clean_data

    except Exception as e:
        logger.error(f"Video Indexer Failed: {e}")

        return {
            "error": [str(e)],
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": [],
        }


# Node 2 : Compliance Auditor 

def audio_content_node(state:VideoAuditState)-> Dict[str, Any]:
    '''
    Performs retriveal augemeted generation to aduit the contetnt - brand videos
    '''
    logger.info("---- [Node: Auditor] querying knowlege base & LLM")
    transcript = state.get("transcript", "")
    if not transcript:
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (NO Transcript)"
        }
    # intialize the clients
    llm = AzureChatOpenAI(
         azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
         openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
         temperature = 0.0
    )    

    embedding = AzureOpenAIEmbeddings(
        azure_deployment = "text-embedding-3-small",
        openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    )

    vector_store = AzureSearch(
        azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key = os.getenv("AZURE_SEARCH_API_KEY"),
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
        embedding_function = embedding.embed_query
    )

    # RAG Retrieval 
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=3)
    retrieved_rules ="\n\n".join([doc.page_content for doc in docs])

    # system prompt 



  
    # --- UPDATED PROMPT WITH STRICT SCHEMA ---
    system_prompt = f"""
    You are a Senior Brand Compliance Auditor.
    
    OFFICIAL REGULATORY RULES:
    {retrieved_rules}
    
    INSTRUCTIONS:
    1. Analyze the Transcript and OCR text below.
    2. Identify ANY violations of the rules.
    3. Return strictly JSON in the following format:
    
    {{
        "compliance_results": [
            {{
                "category": "Claim Validation",
                "severity": "CRITICAL",
                "description": "Explanation of the violation..."
            }}
        ],
        "status": "FAIL", 
        "final_report": "Summary of findings..."
    }}

    If no violations are found, set "status" to "PASS" and "compliance_results" to [].
    """

    user_message = f"""
    VIDEO METADATA: {state.get('video_metadata', {})}
    TRANSCRIPT: {transcript}
    ON-SCREEN TEXT (OCR): {ocr_text}
    """

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        # --- FIX: Clean Markdown if present (```json ... ```) ---
        content = response.content
        if "```" in content:
            # Regex to find JSON inside code blocks
            content = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL).group(1)
            
        audit_data = json.loads(content.strip())
        
        return {
            "compliance_results": audit_data.get("compliance_results", []),
            "final_status": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "No report generated.")
        }

    except Exception as e:
        logger.error(f"System Error in Auditor Node: {str(e)}")
        # Log the raw response to see what went wrong
        logger.error(f"Raw LLM Response: {response.content if 'response' in locals() else 'None'}")
        return {
            "errors": [str(e)],
            "final_status": "FAIL"
        }