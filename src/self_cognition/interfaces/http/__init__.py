"""Local HTTP interface for the application services."""

from self_cognition.interfaces.http.server import create_app, create_server

__all__ = ["create_app", "create_server"]
