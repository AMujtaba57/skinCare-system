import streamlit as st


def register():
    with st.form(key='form1'):
        col1, col2 = st.beta_columns(2)
        fname = col1.text_input("First Name: ")
        lname = col2.text_input("Last Name:")
        user_email = st.text_input("Email: ")
        user_pass = st.text_input("Password: ")
        re_pass = st.text_input("Confirm Password:")
        signup_btn = st.form_submit_button(label="Register")
