import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd
import streamlit as st

from ai import get_answer
from search import HybridSearch
from pdf import process_pdf

PROJECTS_DIR = os.path.expanduser("~/esg/projects")

def list_projects():
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def create_project(project_name):
    path = os.path.join(PROJECTS_DIR, project_name, "uploaded_files")
    os.makedirs(path, exist_ok=True)
    return path

def delete_project(project_name):
    shutil.rmtree(os.path.join(PROJECTS_DIR, project_name))

def list_uploaded_files(project_name):
    files_directory = os.path.join(PROJECTS_DIR, project_name, "uploaded_files")
    files = []
    for filename in os.listdir(files_directory):
        if filename.endswith(".json"):
            with open(os.path.join(files_directory, filename), 'r') as f:
                metadata = json.load(f)
            pdf_filename = filename.replace(".json", ".pdf")
            pdf_path = os.path.join(files_directory, pdf_filename)
            size = os.path.getsize(pdf_path) / (1024 * 1024)  # size in MB
            files.append({"title": metadata['title'], "size": round(size, 2), "filename": pdf_filename})
    files.sort(key=lambda x: x['title'])  # Sort files by title
    return files

def save_uploaded_file(uploaded_file, title, description, project_name):
    directory = os.path.join(PROJECTS_DIR, project_name, "uploaded_files")
    file_hash = hashlib.sha256()
    uploaded_file.seek(0)
    for chunk in uploaded_file:
        file_hash.update(chunk)
    
    file_path = os.path.join(directory, file_hash.hexdigest() + ".pdf")
    if os.path.exists(file_path):
        return False  # File already exists
    else:
        with open(file_path, "wb") as f:
            uploaded_file.seek(0)
            f.write(uploaded_file.read())
        metadata_path = os.path.join(directory, file_hash.hexdigest() + ".json")
        with open(metadata_path, "w") as meta_file:
            json.dump({"title": title, "description": description}, meta_file)

        with st.spinner('Processing PDF...'):
            page_chunks = process_pdf(file_path)
        st.success('PDF processed successfully!')
        hs = HybridSearch(project_name)
        hs.add_document(title,page_chunks)
        return hs
        

def generate_report(project_name,searcher=None) -> pd.DataFrame:
    if not searcher:
        searcher = HybridSearch(project_name)
        print(searcher.vector_search.doc_list)
    gap_df = pd.read_csv(os.path.expanduser('~/esg/esg_gap.csv'))
    print(gap_df.head())
    gap_df.columns = [x.strip() for x in gap_df.columns]

    questions = gap_df['Description'].tolist()

    gap_df['Answer'] = ''
    gap_df['Score'] = ''
    gap_df['Document'] = ''
    gap_df['Page'] = ''
    gap_df['Table'] = ''

    progress_bar = st.progress(0.0)
    for i,q in enumerate(questions):
        print(f'{i}/{len(questions)}')  
        progress_bar.progress(i/len(questions),text=f'question: {q}')
        results = searcher.search(q)
        answer = get_answer('\n'.join(results),q)
        gap_df.iloc[i,'Answer'] = answer['answer']
        gap_df.iloc[i,'Score'] = str(answer['score'])
        gap_df.iloc[i,'Document'] = str(answer['document'])
        gap_df.iloc[i,'Page'] = str(answer['page'])
        gap_df.iloc[i,'Table'] = str(answer['table'])

    return gap_df

if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

# Sidebar for navigation
st.sidebar.title("AI-mily")

action = st.sidebar.radio("Choose an action:", ["Create New Project", "Select Existing Project", "Delete Project"])

if action == "Create New Project":
    project_name = st.sidebar.text_input("Enter new project name")
    if st.sidebar.button("Create Project"):
        if project_name in list_projects():
            st.sidebar.error("Project already exists.")
        elif not project_name:
            st.sidebar.error("Project name cannot be empty.")
        else:
            create_project(project_name)
            st.sidebar.success(f"Project '{project_name}' created!")
 
elif action == "Select Existing Project":
    project_name = st.sidebar.selectbox("Select a project", list_projects())
    if project_name:

        uploaded_files = list_uploaded_files(project_name)
        keyed_uploaded_files = {x['title']: x for x in uploaded_files}

        if uploaded_files:
            st.subheader("Exisiting Project Files")
            selected_uploaded_file = st.selectbox("Select a file", keyed_uploaded_files.keys())
            if selected_uploaded_file and selected_uploaded_file != "Select a file":
                file_path = os.path.join(PROJECTS_DIR, project_name, "uploaded_files", keyed_uploaded_files[selected_uploaded_file]['filename'])
                with open(file_path, "rb") as file:
                    st.download_button(label='Download', data=file, file_name=selected_uploaded_file + '.pdf', mime='application/pdf')
        st.subheader("Add Reference Files")
        if 'upload_key' not in st.session_state:
            st.session_state.upload_key = 0  # Initialize session state for upload key
        uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'], key=f"file_uploader_{st.session_state.upload_key}")
        if uploaded_file:
            title = st.text_input("Enter title for the PDF", key="file_title")
            description = st.text_area("Enter description for the PDF (optional)", key="file_desc")
            if st.button("Upload File"):
                if not title:
                    st.error("Title cannot be empty")
                else:
                    searcher = save_uploaded_file(uploaded_file, title, description, project_name)
                    if not searcher:
                        st.error(f"{title} already in database")
                    else:
                        st.success(f"Saved file: {title} in project '{project_name}'")
                        uploaded_file = None
                        st.session_state.upload_key += 1  # Update key to reset the file_uploader
                        st.rerun()
        
        st.subheader("Generate Report")
        if st.button("Create Report"):
            st.write("Generating report...")
            df = generate_report(project_name)
            if df is not None:
                st.success("Report generated successfully!")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Report as CSV",
                    data=csv,
                    file_name=f"{project_name}_report.csv",
                    mime='text/csv'
                )
            else:
                st.error("Failed to generate report.")
    else:
        st.warning("No projects found.")

elif action == "Delete Project":
    project_name = st.sidebar.selectbox("Select a project to delete", list_projects())
    if project_name:
        if st.sidebar.button("Delete Project"):
            delete_project(project_name)
            st.sidebar.success(f"Project '{project_name}' deleted!")
            st.rerun()
    else:
        st.warning("No projects found.")
