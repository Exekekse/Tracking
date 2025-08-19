"""Detection data container used across the project."""

from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a single detection bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int
    cls: int
    conf: float

    @property
    def cx(self) -> int:
        """Horizontal centre of the box."""
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        """Vertical centre of the box."""
        return (self.y1 + self.y2) // 2

    @property
    def wh(self) -> tuple[int, int]:
        """Width and height of the bounding box."""
        return self.x2 - self.x1, self.y2 - self.y1

