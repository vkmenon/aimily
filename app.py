import hashlib
import json
import os
import pickle
import shutil

import numpy as np
import pandas as pd
import streamlit as st

from ai import get_answer
from docstore import Document, DocStore

PROJECTS_DIR = os.path.expanduser("~/esg/projects")

def list_projects():
    return [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]

def create_project(project_name):
    path = os.path.join(PROJECTS_DIR, project_name, "existing_files")
    os.makedirs(path, exist_ok=True)
    return path

def delete_project(project_name):
    shutil.rmtree(os.path.join(PROJECTS_DIR, project_name))

def list_existing_files(project_name):
    files_directory = os.path.join(PROJECTS_DIR, project_name, "existing_files")
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

def get_project_docstore(project_name):
    docstore_path = os.path.join(PROJECTS_DIR, project_name, "docstore.pkl")
    return DocStore(docstore_path)

def save_uploaded_file(uploaded_file, title, project_name, description='',rechunk_num_pages=0, rechunk_overlap=0):
    directory = os.path.join(PROJECTS_DIR, project_name, "existing_files")
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
            document = Document(file_path, title, situate_context=False)
        st.success('PDF processed successfully!')
        return document
        

def generate_report(docstore) -> pd.DataFrame:
    gap_df = pd.read_csv(os.path.expanduser('~/esg/esg_gap.csv'))
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
        page_results = docstore.hybrid_search(q)
        context = '\n'.join([p.embed_text for p in page_results])
        answer = get_answer(context,q)
        gap_df.loc[i,'Answer'] = answer['answer']
        gap_df.loc[i,'Score'] = str(answer['score'])
        gap_df.loc[i,'Document'] = str(answer['document'])
        gap_df.loc[i,'Page'] = str(answer['page'])
        gap_df.loc[i,'Table'] = str(answer['table'])

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

        existing_files = list_existing_files(project_name)
        keyed_existing_files = {x['title']: x for x in existing_files}

        if existing_files:
            st.subheader("Exisiting Project Files")
            selected_uploaded_file = st.selectbox("Select a file", keyed_existing_files.keys())
            if selected_uploaded_file and selected_uploaded_file != "Select a file":
                file_path = os.path.join(PROJECTS_DIR, project_name, "existing_files", keyed_existing_files[selected_uploaded_file]['filename'])
                with open(file_path, "rb") as file:
                    st.download_button(label='Download', data=file, file_name=selected_uploaded_file + '.pdf', mime='application/pdf')
        
        docstore = get_project_docstore(project_name)

        st.subheader("Add Reference Files")
        if 'upload_key' not in st.session_state:
            st.session_state.upload_key = 0  # Initialize session state for upload key
        uploaded_file = st.file_uploader("Choose PDF file", type=['pdf'], key=f"file_uploader_{st.session_state.upload_key}")
        if uploaded_file:
            title = st.text_input("Enter title for the PDF", key="file_title")
            if st.button("Upload File"):
                if not title:
                    st.error("Title cannot be empty")
                else:
                    document = save_uploaded_file(uploaded_file, title, project_name)
                    if not document:
                        st.error(f"{title} already in database")
                    else:
                        with st.spinner('Adding document to AI database...'):
                            docstore.add_document(document)
                        st.success(f"Saved file: {document.title} in project '{project_name}'")
                        uploaded_file = None
                        st.session_state.upload_key += 1  # Update key to reset the file_uploader
                        st.rerun()
        
        if docstore.title_index.keys():
            st.subheader("Generate Report")
            if st.button("Create Report"):
                st.write("Generating report...")
                df = generate_report(docstore)
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
