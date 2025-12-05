from typing import List, Optional

from pydantic import BaseModel, Field


class UserState(BaseModel):
    stage_dropped: Optional[str] = Field(None, description="Stage where the user dropped (e.g., VKYC, OTP)")
    error_codes: List[str] = Field(default_factory=list)
    device_type: Optional[str] = "Unknown"
    timestamp: Optional[str] = None
    language: str = Field(default="english", description="Preferred response language")


class ChatRequest(BaseModel):
    user_state: UserState
    query: str


class NudgeRequest(BaseModel):
    user_state: UserState
    nudge_type: str
    language: str = "english"

