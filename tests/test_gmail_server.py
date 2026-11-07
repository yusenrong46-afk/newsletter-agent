"""The MCP server: exactly two tools, clean stdout, and careful MIME handling."""

import asyncio
import json
import os
import subprocess
import sys

from helpers import message
from newsletter_agent.demo import SyntheticGmail, encode
from newsletter_agent.gmail_client import GmailToolError, open_gmail_client
from newsletter_agent.gmail_server import read_message


def test_real_subprocess_handshake_exposes_exactly_two_tools():
    async def go():
        async with open_gmail_client() as gmail:
            tools = await gmail.session.list_tools()
            return {tool.name for tool in tools.tools}

    assert asyncio.run(go()) == {"search_emails", "read_email"}


def test_server_stdout_contains_only_protocol_messages():
    request = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}}}
    result = subprocess.run(
        [sys.executable, "-m", "newsletter_agent", "mcp-server"], env=dict(os.environ),
        input=json.dumps(request) + "\n", capture_output=True, text=True, timeout=30,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines
    assert json.loads(lines[0])["id"] == 1
    assert all(json.loads(line) for line in lines)


def test_tool_errors_keep_a_stable_code_without_echoing_details():
    error = GmailToolError.from_tool_text("Error executing tool read_email: not_found: Gmail no longer has this message.")
    assert error.code == "not_found"
    assert GmailToolError.from_tool_text("something unexpected with secrets").code == "gmail_error"
    assert "secrets" not in str(GmailToolError.from_tool_text("something unexpected with secrets"))


def read(msg, **arguments):
    return read_message(SyntheticGmail([msg]), msg["id"], **arguments)


def test_html_only_message_becomes_text_without_scripts():
    html = "<html><head><title>no</title><style>.x{}</style></head><body><p>Fish &amp; chips</p>" \
           "<script>steal()</script><img src='https://track.example/p.gif'><p>Second</p></body></html>"
    result = read(message("h", payload={"mimeType": "text/html", "body": {"data": encode(html)}}))
    assert result["body"] == "Fish & chips\nSecond"
    assert result["body_status"] == "available"


def test_nested_mime_prefers_plain_text_decodes_charset_and_skips_attachments():
    msg = message("n", payload={"mimeType": "multipart/mixed", "parts": [
        {"mimeType": "multipart/alternative", "parts": [
            {"mimeType": "text/plain", "headers": [{"name": "Content-Type", "value": 'text/plain; charset="iso-8859-1"'}],
             "body": {"data": encode("Café news", "iso-8859-1")}},
            {"mimeType": "text/html", "body": {"data": encode("<p>HTML version</p>")}},
        ]},
        {"mimeType": "text/plain", "filename": "notes.txt", "body": {"data": encode("ATTACHMENT TEXT")}},
    ]})
    assert read(msg)["body"] == "Café news"


def test_truncated_and_unavailable_bodies_stay_explicit():
    result = read(message("l", body="x" * 50), max_chars=10)
    assert (result["body_status"], len(result["body"]), result["body_truncated"]) == ("truncated", 10, True)

    image_only = message("i", payload={"mimeType": "image/png", "body": {"attachmentId": "A"}})
    result = read(image_only)
    assert (result["body_status"], result["body_available"]) == ("unavailable", False)

    not_inline = message("o", payload={"mimeType": "multipart/mixed", "parts": [
        {"mimeType": "text/plain", "body": {"data": encode("start")}},
        {"mimeType": "text/plain", "body": {"attachmentId": "B"}},
    ]})
    assert read(not_inline)["body_status"] == "truncated"


def test_metadata_mode_returns_headers_case_insensitively_and_no_body():
    msg = message("m", headers={"LIST-ID": "<Upper.Example.COM>", "list-unsubscribe": "<mailto:x@y.example>"})
    result = read(msg, mode="metadata")
    assert result["headers"]["list-id"] == "<Upper.Example.COM>"
    assert result["headers"]["list-unsubscribe"]
    assert (result["body"], result["body_status"]) == ("", "not_requested")
