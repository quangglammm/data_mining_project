"""Yield class enumeration."""

from enum import Enum


class YieldClass(str, Enum):
    """Enumeration for rice yield classification."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

    def to_vietnamese(self) -> str:
        """Convert to Vietnamese label."""
        mapping = {
            YieldClass.HIGH: "Năng suất Cao",
            YieldClass.MEDIUM: "Năng suất Trung bình",
            YieldClass.LOW: "Năng suất Thấp",
        }
        return mapping[self]

