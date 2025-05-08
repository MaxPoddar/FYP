import streamlit as st
from main import start_server, check_server, prepare_chatbot
from LLM import process_user_input

# Define available user roles
roles = [
    "CEO",
    "CTO",
    "VP of Information Technology",
    "IT Director",
    "SW Manager",
    "IT Administrator"
]

# Initialize chat messages in session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'sender': 'bot', 'text': 'Hello there!'}
    ]

# First-time setup: start/check server and prepare chatbot
if 'chatbot_initialized' not in st.session_state:
    st.session_state.server_status = start_server()
    st.session_state.server_check = check_server()

    gen = prepare_chatbot()
    setup_msgs = []
    try:
        while True:
            setup_msgs.append(next(gen))
    except StopIteration as stop:
        (
            st.session_state.machine_ids,
            st.session_state.model,
            st.session_state.index,
            st.session_state.sentences,
            st.session_state.send_prompt,
            st.session_state.questions
        ) = stop.value

    st.session_state.setup_msgs = setup_msgs
    st.session_state.chatbot_initialized = True

# Sidebar: Chats, server status, setup logs
with st.sidebar:
    st.title("Chats")
    if st.button("New Chat"):
        st.session_state.messages = [ {'sender': 'bot', 'text': 'Hello there!'} ]

    st.subheader("Server Status")
    st.write(st.session_state.server_status)
    st.write(st.session_state.server_check)

    st.subheader("Chatbot Setup Logs")
    for log in st.session_state.setup_msgs:
        st.write(log)

# Main layout: role dropdown and chat area
col1, col2 = st.columns([8, 2])
with col2:
    st.selectbox("Select Role", roles, key='user_role')

# Clear chat history when role changes
if 'prev_user_role' not in st.session_state:
    st.session_state.prev_user_role = st.session_state.user_role
elif st.session_state.prev_user_role != st.session_state.user_role:
    st.session_state.messages = [ {'sender': 'bot', 'text': 'Hello there!'} ]
    st.session_state.prev_user_role = st.session_state.user_role

st.title("Chat with Bot")

# Display stored messages
for msg in st.session_state.messages:
    st.chat_message(msg['sender']).write(msg['text'])

# Chat input
user_input = st.chat_input("Type a message...")

if user_input:
    # Display and store user message immediately
    st.chat_message('user').write(user_input)
    st.session_state.messages.append({'sender': 'user', 'text': user_input})

    # Stream bot response within a single chat bubble
    full_response = ""
    with st.chat_message('bot'):
        placeholder = st.empty()
        for chunk in process_user_input(
            user_question=user_input,
            user_role=st.session_state.user_role,
            machine_ids=st.session_state.machine_ids,
            model=st.session_state.model,
            index=st.session_state.index,
            sentences=st.session_state.sentences,
            send_prompt=st.session_state.send_prompt,
            questions=st.session_state.questions
        ):
            full_response += chunk
            placeholder.markdown(full_response)

    # Save full bot message to history
    st.session_state.messages.append({'sender': 'bot', 'text': full_response})
