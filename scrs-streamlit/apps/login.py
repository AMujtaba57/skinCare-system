import streamlit as st

def login():
    with st.form(key='form1'):
        user_email = st.text_input("Email: ")
        user_pass = st.text_input("Password: ")
        login_btn = st.form_submit_button(label="Login")
        try:
            if user_email != "" and user_pass !="":
                user = authe.sign_in_with_email_and_password(user_email, user_pass)
                if user:
                    st.success(user_email)
                else:
                    st.write("Invalid Credential")
        except:
            st.write("Invalid Credential")
