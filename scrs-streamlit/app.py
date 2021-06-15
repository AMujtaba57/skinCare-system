import streamlit as st
from multiapp import MultiApp
from apps import analyse, care_routine, login, signup, step # import your app modules here

app = MultiApp()
app1 = MultiApp()

st.title("SkinCare Recommender System")
st.markdown("<br><br>", unsafe_allow_html=True)
user = True
page = ""

if user:
    page = "Home"
    app.add_app("Analyse", analyse.layout)
    app.add_app("Care Routine", care_routine.care_routine)
    app.add_app("Past Cases", analyse.layout)
    app.add_app("Trigger Tracker", analyse.layout)


else:
    app.add_app("Login", login.login)
    app.add_app("Signup", signup.register)
    page = "Credential"
# The main app
app.run(page)
