"""Services for OCI GenAI integration."""

from .generation import generate_oci_non_stream, generate_oci_stream

__all__ = ["generate_oci_non_stream", "generate_oci_stream"]
