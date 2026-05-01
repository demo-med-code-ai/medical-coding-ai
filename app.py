

import streamlit as st
from datetime import datetime
from transformers import pipeline

import pandas as pd

st.info("This is a beta AI-assisted coding tool. Do NOT enter real patient information. Results are for educational/demo purposes only.")

st.set_page_config(page_title="Medical Coding AI", layout="wide")

st.title("Medical Coding AI")
st.caption("AI-assisted ICD-10 coding tool (Beta)")

st.warning("Demo only. Do not enter real patient information.")
users = {
    "demo": "demo123"
}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.stop()

st.sidebar.write("Logged in as:")
st.sidebar.success(st.session_state.username)
paid_users = ["demo"]

if st.session_state.username not in paid_users:
    st.error("Subscription required.")
    st.write("This account does not have access to the coding tool yet.")
    st.write("Demo plan: $49/month")
    st.stop()

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

@st.cache_resource
def load_ai_model():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

ai_model = load_ai_model()

code_rules = {
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications", "High"),
    "diabetic neuropathy": ("E11.40", "Type 2 diabetes mellitus with diabetic neuropathy, unspecified", "High"),
    "hypertension": ("I10", "Essential hypertension", "High"),
    "high blood pressure": ("I10", "Essential hypertension", "High"),
    "chest pain": ("R07.9", "Chest pain, unspecified", "Medium"),
    "shortness of breath": ("R06.02", "Shortness of breath", "Medium"),
    "asthma": ("J45.909", "Unspecified asthma, uncomplicated", "Medium"),
    "bronchitis": ("J20.9", "Acute bronchitis, unspecified", "Medium"),
    "copd": ("J44.9", "Chronic obstructive pulmonary disease, unspecified", "Medium"),
    "pneumonia": ("J18.9", "Pneumonia, unspecified organism", "Medium"),
    "cough": ("R05.9", "Cough, unspecified", "Medium"),
    "fever": ("R50.9", "Fever, unspecified", "Medium"),
    "headache": ("R51.9", "Headache, unspecified", "Medium"),
    "migraine": ("G43.909", "Migraine, unspecified, not intractable, without status migrainosus", "Medium"),
    "back pain": ("M54.50", "Low back pain, unspecified", "Medium"),
    "neck pain": ("M54.2", "Cervicalgia", "Medium"),
    "knee pain": ("M25.569", "Pain in unspecified knee", "Medium"),
    "shoulder pain": ("M25.519", "Pain in unspecified shoulder", "Medium"),
    "abdominal pain": ("R10.9", "Unspecified abdominal pain", "Medium"),
    "nausea": ("R11.0", "Nausea", "Medium"),
    "vomiting": ("R11.10", "Vomiting, unspecified", "Medium"),
    "diarrhea": ("R19.7", "Diarrhea, unspecified", "Medium"),
    "uti": ("N39.0", "Urinary tract infection, site not specified", "Medium"),
    "urinary tract infection": ("N39.0", "Urinary tract infection, site not specified", "Medium"),
    "depression": ("F32.A", "Depression, unspecified", "Medium"),
    "anxiety": ("F41.9", "Anxiety disorder, unspecified", "Medium"),
    "obesity": ("E66.9", "Obesity, unspecified", "Medium"),
    "fatigue": ("R53.83", "Other fatigue", "Medium"),
    "dizziness": ("R42", "Dizziness and giddiness", "Medium"),
    "allergic rhinitis": ("J30.9", "Allergic rhinitis, unspecified", "Medium"),
}

    
note = st.text_area("Enter clinical note:", height=200)

if st.button("Suggest Codes"):
    if note:
        candidate_labels = [
            f"{code} - {desc}"
            for keyword, (code, desc, conf) in code_rules.items()
        ]

        ai_result = ai_model(
            note,
            candidate_labels,
            multi_label=True
        )

        results = []

        for label, score in zip(ai_result["labels"], ai_result["scores"]):
            if score >= 0.35:
                parts = label.split(" - ")
                code = parts[0]
                desc = parts[1]

                if score >= 0.70:
                    conf = "High"
                else:
                    conf = "Medium"

                results.append((code, desc, conf, round(score, 2)))

        results = results[:5]
        
        st.session_state.results = results
        st.session_state.note = note

if "results" in st.session_state:
    results = st.session_state.results

    st.subheader("Top Suggested Codes")
   
    for code, desc, conf, score in results:
        st.write(f"{code} - {desc} ({conf}, AI score: {score})")
    
    # KEEP selection persiste
    if "approved" not in st.session_state:
        st.session_state.approved = []

    approved = st.multiselect(
                "Select codes to approve:",
                [f"{code} - {desc}" for code, desc, conf, score in results],
                default=st.session_state.approved
            )    

    # Save selection so it doesn't disappear
    st.session_state.approved = approved

    if st.button("Save Review"):
       if not approved:
          st.error("Please select at least one code.")
       else:
           import pandas as pd
           from datetime import datetime
      
           data = {
               "User": st.session_state.username,
               "Note": st.session_state.note,
               "Approved Codes": ", ".join(approved),
               "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
           }
              
           df = pd.DataFrame([data])

           try:
              existing = pd.read_csv("history.csv")
              df = pd.concat([existing, df], ignore_index=True)
           except:
              pass
              
           df.to_csv("history.csv", index=False)

           st.success("Review saved successfully.")

import os

st.subheader("Early Access")

st.warning("Limited beta access - early users will get discounted pricing.")

email = st.text_input("Want full access to faster, more accurate coding? Join the beta:")

if st.button("Request full access"):
    if email.strip() == "":
        st.error("Please enter your email.")
    else:
        import pandas as pd
        from datetime import datetime

        data = {
            "Email": email,
            "User": st.session_state.username,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        df = pd.DataFrame([data])

        try:
            existing = pd.read_csv("access_requests.csv")
            df = pd.concat([existing, df], ignore_index=True)
        except:
            pass

        df.to_csv("access_requests.csv", index=False)

        st.success("Request received. Thank you!")

st.subheader("Audit History")

# Restrict demo users
if st.session_state.username == "demo":
    st.warning("Audit history is disabled for demo users.")
else:
    import os

    if os.path.exists("history.csv"):
        history = pd.read_csv("history.csv")
        st.dataframe(history, use_container_width=True)
    else:
        st.write("No history yet.")
