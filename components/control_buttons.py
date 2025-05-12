"""
Control buttons for process management and UI control.
"""
import streamlit as st

def create_control_buttons(location="main", process_id=None):
    """
    Create a minimal stop button for process management.

    Parameters:
    -----------
    location : str
        Where the button is being placed ('main', 'sidebar', or specific component name)
    process_id : str, optional
        Identifier for the specific process this button controls

    Returns:
    --------
    dict
        Dictionary with button states (clicked or not)
    """
    # Initialize return dictionary
    button_states = {
        'stop': False
    }

    # Create a container for the button
    if location == "sidebar":
        container = st.sidebar
    else:
        container = st

    # Create a small container for the stop button
    # Use a small column to position the button on the right
    _, right_col = container.columns([9, 1])

    # Create a small circular stop button
    with right_col:
        # Custom CSS for the stop button to make it circular and minimal
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] button[kind="secondary"] {
                border-radius: 50%;
                width: 32px;
                height: 32px;
                padding: 0px;
                font-size: 14px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-left: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Create the stop button
        if process_id:
            stop_help = f"Stop the current {process_id} process"
            stop_key = f"stop_button_{location}_{process_id}"
        else:
            stop_help = "Stop the currently running background process"
            stop_key = f"stop_button_{location}"

        button_states['stop'] = st.button(
            "⏹️",
            help=stop_help,
            key=stop_key,
            type="secondary"
        )

        if button_states['stop']:
            # Set the process state to stopped
            if process_id and process_id in st.session_state.active_processes:
                st.session_state.active_processes[process_id] = False

            # Show a subtle message
            st.info("Process stopped. You can continue working with the loaded data.")

    return button_states

def check_process_status(process_id):
    """
    Check if a process should continue running.

    Parameters:
    -----------
    process_id : str
        Identifier for the specific process

    Returns:
    --------
    bool
        True if the process should continue, False if it should stop
    """
    # Initialize the process if it doesn't exist
    if process_id not in st.session_state.active_processes:
        st.session_state.active_processes[process_id] = True

    # Return the current status
    return st.session_state.active_processes[process_id]

def start_process(process_id):
    """
    Mark a process as started.

    Parameters:
    -----------
    process_id : str
        Identifier for the specific process
    """
    st.session_state.active_processes[process_id] = True

def stop_process(process_id):
    """
    Mark a process as stopped.

    Parameters:
    -----------
    process_id : str
        Identifier for the specific process
    """
    st.session_state.active_processes[process_id] = False
