"""Audit logging module for production-grade audit trails."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from config.logging_config import get_logger
from config.settings import MAPPING_AUDIT_FILE

logger = get_logger(__name__)


class AuditLogger:
    """Log mapping operations and decisions for audit trail."""

    def __init__(self, audit_file: Path = MAPPING_AUDIT_FILE):
        """Initialize audit logger."""
        self.audit_file = audit_file
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log a structured audit event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
        }

        try:
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(event) + "\n")
            logger.debug(f"Audit event logged: {event_type}")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def log_hotel_match(
        self,
        hotelbeds_hotel: Dict[str, Any],
        axisdata_hotel: Dict[str, Any] | None,
        confidence: float,
    ):
        """Log hotel matching decision."""
        self.log_event(
            "hotel_match",
            {
                "hotelbeds_id": hotelbeds_hotel.get("hotel_id"),
                "hotelbeds_name": hotelbeds_hotel.get("name"),
                "axisdata_id": axisdata_hotel.get("hotel_id") if axisdata_hotel else None,
                "axisdata_name": axisdata_hotel.get("name") if axisdata_hotel else None,
                "matched": axisdata_hotel is not None,
                "confidence": confidence,
            },
        )

    def log_room_match(
        self,
        hotel_id: str,
        hotelbeds_room: Dict[str, Any],
        enriched_room: Dict[str, Any] | None,
        confidence: float,
    ):
        """Log room matching decision."""
        self.log_event(
            "room_match",
            {
                "hotel_id": hotel_id,
                "hotelbeds_room_code": hotelbeds_room.get("code"),
                "hotelbeds_room_name": hotelbeds_room.get("name"),
                "matched_room_code": enriched_room.get("code") if enriched_room else None,
                "matched": enriched_room is not None,
                "confidence": confidence,
            },
        )

    def log_pipeline_start(self, metadata: Dict[str, Any]):
        """Log pipeline execution start."""
        self.log_event("pipeline_start", metadata)

    def log_pipeline_complete(self, metadata: Dict[str, Any]):
        """Log pipeline execution completion."""
        self.log_event("pipeline_complete", metadata)

    def log_error(self, error_type: str, details: Dict[str, Any]):
        """Log processing error."""
        self.log_event("error", {"error_type": error_type, "details": details})
