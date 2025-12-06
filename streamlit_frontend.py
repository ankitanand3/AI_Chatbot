from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from pydantic import config
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}
# session state = dictionary that does not erase until page is refreshed manually

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# Loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')      

if user_input:
    # First add the message to message_hostory
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = chatbot.invoke({'messages': [HumanMessage(content = user_input)]}, config = CONFIG)
    ai_message = response['messages'][-1].content
    # First add the message to message_history
    st.session_state['message_history'].append({'role': 'assisstant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)