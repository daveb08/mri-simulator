"""
Token verifier for the MRI Simulator Streamlit app.

Replace the existing static-token check in your Streamlit app with this module.
The token is short-lived (10 min default) and signed by the MRI Mastery API
server using HMAC-SHA256, so it cannot be reused after expiry and cannot be
forged without the shared secret.

Setup
-----
1. In Streamlit Cloud, set a secret named SIMULATOR_SIGNING_SECRET to the
   SAME value used by the MRI Mastery API server.
2. Drop this file into your Streamlit app's repo (next to streamlit_app.py).
3. In your main app file, replace the old token check with:

       import streamlit as st
       from streamlit_token_verifier import require_valid_token

       user_id = require_valid_token()  # halts the app on failure
       # ...rest of your app...

The function will st.stop() the app and show an error if the token is
missing, malformed, expired, or the signature doesn't match.
"""

from __future__ import annotations

import base64
import hmac
import hashlib
import json
import os
import time
from typing import Optional

import streamlit as st


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def verify_simulator_token(token: str, secret: str) -> Optional[dict]:
    """
    Returns the decoded payload dict ({"u": user_id, "e": expiry_epoch})
    if the token is valid and unexpired. Returns None otherwise.
    """
    if not token or not secret:
        return None
    try:
        payload_b64, sig_hex = token.split(".", 1)
    except ValueError:
        return None

    expected_sig = hmac.new(
        secret.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, sig_hex):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None

    exp = payload.get("e")
    if not isinstance(exp, (int, float)) or time.time() >= exp:
        return None

    return payload


def require_valid_token() -> str:
    """
    Reads ?token=... from the Streamlit query params, verifies it, and
    returns the user id on success. Halts the app on failure.
    """
    secret = os.environ.get("SIMULATOR_SIGNING_SECRET") or st.secrets.get(
        "SIMULATOR_SIGNING_SECRET", ""
    )
    if not secret:
        st.error("Simulator is misconfigured (missing signing secret).")
        st.stop()

    token = st.query_params.get("token", "")
    if isinstance(token, list):
        token = token[0] if token else ""

    payload = verify_simulator_token(token, secret)
    if not payload:
        st.error(
            "Access denied. This simulator link is invalid or has expired. "
            "Please return to MRI Mastery and re-open the simulator."
        )
        st.stop()

    return str(payload["u"])
