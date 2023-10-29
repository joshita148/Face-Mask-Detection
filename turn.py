import logging
import os
import streamlit as st
from twilio.rest import Client

logger = logging.getLogger(__name__)



@st.cache_data
def get_ice_servers():

    try:
        account_sid = st.secrets["account_sid"]
        auth_token = st.secrets["auth_token"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers
