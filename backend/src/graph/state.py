from typing import TypedDict, List, Dict, Any

class VideoAuditState(TypedDict):
    video_url: str
    video_id: str
    transcript: str
    ocr_text: List[str]
    video_metadata: Dict[str, Any]
    compliance_results: List[Dict[str, Any]]
    final_status: str
    final_report: str
    error: List[str]
