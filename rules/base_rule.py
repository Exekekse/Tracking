"""Base class for rule plug‑ins."""

from detection import Detection


class Rule:
    name = "BaseRule"

    def accept(self, det: Detection, frame) -> bool:  # pragma: no cover - to be overridden
        """Return ``True`` to keep the detection, ``False`` to discard it."""
        return True

