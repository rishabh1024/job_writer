from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class TavilyQuerySet(BaseModel):
    query1: Optional[List[str]] = Field(default=None, description="First search query and its rationale, e.g., ['query text']")
    query2: Optional[List[str]] = Field(default=None, description="Second search query and its rationale")
    query3: Optional[List[str]] = Field(default=None, description="Third search query and its rationale")
    query4: Optional[List[str]] = Field(default=None, description="Fourth search query and its rationale")
    query5: Optional[List[str]] = Field(default=None, description="Fifth search query and its rationale")

    @field_validator("query1", "query2", "query3", "query4", "query5", mode="after")
    @classmethod
    def ensure_len_two(cls, v):
        """Ensure each provided query list contains exactly one strings: [query]."""
        if v is not None:  # Only validate if the list is actually provided
            if len(v) != 1:
                # Updated error message for clarity
                raise ValueError("Each query list, when provided, must contain exactly one string: the query text.")
        return v
