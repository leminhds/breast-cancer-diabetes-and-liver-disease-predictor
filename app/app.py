import streamlit as st
import awesome_streamlit as ast

import pages.breast_cancer
import pages.diabetes
import pages.liver

PAGES = {
    'Breast Cancer': pages.breast_cancer,
    'Diabetes': pages.diabetes,
    'Liver Disease': pages.liver
}

sidebar_options = st.sidebar.selectbox("Check for:", list(PAGES.keys()))

page = PAGES[sidebar_options]

with st.spinner(f"Loading {sidebar_options} ..."):
    ast.shared.components.write_page(page)
