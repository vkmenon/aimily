import json
import random
import camelot

from multiprocessing import Pool
from collections import Counter
from difflib import SequenceMatcher
from PyPDF2 import PdfReader

from ai import situate_context

def overlap_score(str1, str2):
    if not str1 or not str2:
        return 0.0
    def longest_common_substring(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_length = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_length = max(max_length, dp[i][j])
        
        return max_length
    
    longest_overlap = longest_common_substring(str1, str2)
    min_length = min(len(str1), len(str2))
    
    return longest_overlap / min_length

def read_pdf_pages(pdf_path):
    reader = PdfReader(pdf_path)

    # Extract text from all pages
    page_chunks = []
    for i,page in enumerate(reader.pages):
        page_chunks.append(page.extract_text())
    return page_chunks

# Attempt to extract tables using a different approach (detecting raw content instead of text extraction)
def enhance_page_tables(page_number,pdf_path):
    table = camelot.read_pdf(pdf_path, pages=str(page_number+1))
    txt = ''
    for i,t in enumerate(table):
        txt += f'\n\n<<START TABLE : {i+1}>>\n\n{t.df.to_string()}\n\n<<END TABLE : {i+1}>>\n\n'
    
    if not len(table):
        txt = ''

    return page_number, txt


def clean_pdf_pages(pdf_path,page_chunks):
    meta_path = pdf_path.replace('.pdf','.json')
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    doc_title = metadata['title']

    # Tokenize each page into lines
    tokenized_pages = [page.split('\n') for page in page_chunks]

    # Create a random sample of page numbers
    sampled_page_numbers = random.sample(range(len(tokenized_pages)), int(0.2 * len(tokenized_pages)))
    sampled_page_numbers = sorted(sampled_page_numbers)

    print(f'Sampled page numbers: {sampled_page_numbers}')

    # Find common lines across all pages with 90% coverage and 90% similarity
    common_lines = Counter()
    for i,page_i in enumerate(sampled_page_numbers):
        for j in range(i+1, len(sampled_page_numbers)):
            page_j = sampled_page_numbers[j]

            print(f'Comparing page {page_i} with page {page_j}')

            for line1 in tokenized_pages[page_i]:
                
                found_common = False
                for common_line in common_lines:
                    if SequenceMatcher(None, line1.strip(), common_line.strip()).ratio() > 0.9:
                        common_lines[common_line] += 1
                        found_common = True
                        break
                    if overlap_score(line1.strip(), common_line.strip()) > 0.9:
                        common_lines[common_line] += 1
                        found_common = True
                        break
                
                if found_common:
                    continue

                for line2 in tokenized_pages[page_j]:

                    if line1.startswith('<<'):
                        continue

                    if SequenceMatcher(None, line1, line2).ratio() > 0.9:
                        if line1 in common_lines:
                            common_lines[line1] += 1
                        elif line2 in common_lines:
                            common_lines[line2] += 1
                        else:
                            common_lines[line1] = 1
    
    common_lines = {k for k,v in common_lines.items() if v >= (0.6 * (0.5*len(sampled_page_numbers)*(len(sampled_page_numbers)+1))) and len(k)>5}
    # Remove lines from page_chunks that are similar to common_lines
    for page_index, page in enumerate(page_chunks):
        new_page = []
        for line in page.split('\n'):
            
            similarity = any(SequenceMatcher(None, line.strip(), common_line.strip()).ratio() > 0.90 for common_line in common_lines)
            subset = any(line.strip() in common_line.strip() or common_line.strip() in line.strip() for common_line in common_lines)
            overlap = any(overlap_score(line.strip(), common_line.strip()) > 0.80 for common_line in common_lines)
            
            if not (similarity or subset or overlap):
                new_page.append(line)
                
        new_page = '\n'.join(new_page)
        new_page = f'<<<DOCUMENT : {doc_title}, START PAGE : {page_index+1}>>>\n\n\n{new_page}'
        page_chunks[page_index] = new_page
    
    reader = PdfReader(pdf_path)
    with Pool() as pool:
        results = pool.starmap(enhance_page_tables, [(page_number, pdf_path) for page_number in range(len(reader.pages))])

    for page_number, content in results:
        txt = page_chunks[page_number]
        tmp = [txt,content,f'\n\n\n<<<DOCUMENT : {doc_title}, END PAGE : {page_number+1}>>>']
        page_chunks[page_number] = '\n'.join(tmp)
    
    return page_chunks

def add_context(page_chunks):
    full_text = '\n\n'.join(page_chunks)
    full_text = '\n'.join([line for line in full_text.split('\n') if not (line.startswith('<<<DOCUMENT: ') or line.startswith('<<START TABLE'))])    

    if len(full_text) > 409600:
        full_text = full_text[:409600]

    for page,txt in page_chunks.items():
        context = f'<<START Context of this page within the whole document>>:\n{situate_context(full_text, txt)}<<END Context of this page within the whole document>>\n'
        tmp_txt = txt.split('\n\n\n')
        page_chunks[page] = '\n'.join([tmp_txt[0],context,tmp_txt[1]])
    
    return page_chunks

def process_pdf(pdf_path,add_context=False):
    page_chunks = read_pdf_pages(pdf_path)
    page_chunks = clean_pdf_pages(pdf_path, page_chunks)
    if add_context:
        page_chunks = add_context(page_chunks)
    return page_chunks

def rechunk_pages(page_chunks, num_pages, num_overlapping_pages):
    new_chunks = []
    total_pages = len(page_chunks)
    i = 0

    while i < total_pages:
        end = min(i + num_pages, total_pages)
        new_chunks.append(page_chunks[i:end])
        i = end - num_overlapping_pages

    return new_chunks