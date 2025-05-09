import streamlit as st

from src.state.session import init_session_state

def main():
    init_session_state()
    st.title("Welcome to the D&D Dungeon Master App")   
    st.markdown("Use the sidebar to create characters or start the game session with the DM.")

if __name__ == "__main__":
    main()