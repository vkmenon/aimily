import json
import random
import os
import pickle

import camelot

import numpy as np

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from multiprocessing import Pool

from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import Any, List, Union
import boto3
from urllib.parse import urlparse
import tempfile


class Page:
    """
    A class representing a single page in a document. The Page class handles the text content, metadata, and 
    table extraction for a specific page in a PDF document.

    Attributes:
    raw_text (str): The raw text content of the page.
    page_number (Union[int, List[int]]): The page number(s) of the PDF.
    title (str): The title of the document the page belongs to.
    file_path (str): The file path of the PDF document.
    situated_text (str): Contextually processed text, with surrounding content for semantic analysis.
    cleaned_text (str): Cleaned version of the raw text with common artifacts removed.
    tables (List[str]): A list of tables extracted from the page, represented as strings.
    embed_text (str): Final structured and formatted text of the page, including tables, headers, and footers.
    
    Methods:
    _get_tables(): Extracts tables from the PDF page using Camelot.
    finalize(): Finalizes the page content by combining text, tables, headers, and footers.
    to_dict(): Converts the page's attributes to a dictionary.
    to_json(): Converts the page's attributes to a JSON string.
    """

    def __init__(self, text: str, page_number: Union[int, List[int]], title: str, file_path: str) -> None:
        self.raw_text = text
        self.page_number = page_number
        self.title = title
        self.file_path = file_path
        self.situated_text = None
        self.cleaned_text = None
        self.tables = []
        self.embed_text = None
    
    def _get_tables(self) -> None:
        """
        Extracts tables from the PDF page using Camelot and converts them to string format.
        """
        for table in camelot.read_pdf(self.file_path, pages=str(self.page_number)):
            self.tables.append(table.df.to_string())

    def finalize(self) -> None:
        """
        Finalizes the text content of the page by incorporating tables, headers, and footers into the main text. 
        This method also extracts tables if they have not been extracted yet, and ensures that the final 
        embedded text for the page is structured and formatted appropriately.
        
        Returns:
        self: The Page object after finalization.
        """
        print(f'finalizing page {self.page_number}')
        header = f'<<<DOCUMENT : {self.title}, START PAGE : {self.page_number}>>>\n\n\n'
        footer = f'\n\n\n<<<DOCUMENT : {self.title}, END PAGE : {self.page_number}>>>'

        def table_header(i):
            return f'\n\n<<START TABLE : {i}>>\n\n'
        def table_footer(i):
            return f'\n\n<<END TABLE : {i}>>\n\n'

        if not self.tables:
            self._get_tables()

        table_text = '\n'.join([table_header(i+1) + table + table_footer(i+1) for i,table in enumerate(self.tables)])

        if not self.cleaned_text:
            content = self.raw_text
        
        if not self.situated_text:
            content = self.cleaned_text

        self.embed_text = header + content + table_text + footer

        return self

    def to_dict(self) -> dict:
        """
        Convert the Page object's attributes to a dictionary format.
        
        Returns:
        dict: A dictionary containing the page's attributes such as raw text, page number, title, etc.
        """
        return {
            "raw_text": self.raw_text,
            "page_number": self.page_number,
            "title": self.title,
            "cleaned_text": self.cleaned_text,
            "tables": self.tables,
            "embed_text": self.embed_text
        }
    
    def to_json(self) -> str:
        """
        Convert the Page object's attributes to a JSON string.
        
        Returns:
        str: A JSON string representing the page's attributes.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_path: str) -> 'Page':
        """
        Create a Page object from a JSON file.

        Args:
        json_path (str): The path to the JSON file.

        Returns:
        Page: A Page object created from the JSON file.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        

        required_fields = ['raw_text', 'page_number', 'title']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            
        data_dd = defaultdict(str)
        data_dd.update(data)

        page = cls(
            text=data_dd['raw_text'],
            page_number=data_dd['page_number'],
            title=data_dd['title'],
            file_path=data_dd['file_path']
        )
        page.cleaned_text = data.get('cleaned_text')
        page.tables = data.get('tables', [])
        page.embed_text = data.get('embed_text')
        
        return page

    def __str__(self) -> str:
        if self.embed_text:
            return self.embed_text
        else:
            self.finalize()
            return self.embed_text
    
    def __repr__(self) -> str:
        return f"Page(Document Title: {self.title}, Page Number: {self.page_number})"

class Document:
    """
    A class representing a PDF document that contains multiple pages. The Document class handles 
    reading, cleaning, and processing the pages in a PDF file, and optionally situating the context for semantic searches.

    Attributes:
    title (str): The title of the document.
    file_path (str): The file path of the PDF document.
    pages (List[Page]): A list of Page objects representing the pages in the document.
    rechunked_pages (List[Page]): A list of rechunked pages if the document has been split into chunks.
    
    Methods:
    finalize(n_jobs: int): Uses multiprocessing to finalize all pages in the document.
    rechunk_pages(num_pages: int, overlap: int): Splits the document into chunks based on a specified number of pages per chunk.
    situate_context(): Situates contextual information for each page in the document to assist with semantic search.
    _read_pages(): Reads the PDF and creates Page objects for each page.
    _clean_pages(): Cleans the pages by removing common artifacts such as headers and footers.
    """

    def __init__(self, file_path: str, title: str, situate_context: bool = False) -> None:
        self.title = title
        self.file_path = file_path
        self.pages = list()
        self.rechunked_pages = list()

        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if not self.file_path.endswith('.pdf'):
            raise ValueError("Unsupported file format. Only PDFs are supported right now.")
        
        self._read_pages()
        self._clean_pages()

        if situate_context:
            self.situate_context()
        
    def finalize(self, n_jobs: int = 1) -> None:
        """
        Use multiprocessing to finalize the pages by extracting tables and formatting text.
        
        Args:
        n_jobs (int): Number of parallel jobs to run for multiprocessing.
        """

        if isinstance(n_jobs, int) and n_jobs > 1:
            n_jobs = min(n_jobs, os.cpu_count())
        else:
            n_jobs = 1
        print("Starting multiprocessing with", n_jobs, "jobs.")
        with Pool(n_jobs) as pool:
            # Using a wrapper function for better error handling
            results = pool.map(worker_finalize_page, self.pages)
        self.pages = [result for result in results if result is not None]
        self.pages.sort(key=lambda x: x.page_number)
        print("Completed multiprocessing.")

    def rechunk_pages(self, num_pages: int, overlap: int) -> None:
        """
        Re-chunks the pages of the document based on a specified number of pages per chunk and overlap.
        
        Args:
        num_pages (int): Number of pages per chunk.
        overlap (int): Number of overlapping pages between consecutive chunks.
        """
        pass

    def situate_context(self) -> None:
        """
        Situates contextual information for each from the document to help with semantic search.
        """
        pass

    def _read_pages(self) -> None:
        """
        Reads pages from a PDF document and initializes Page objects.
        """
        if self.file_path.endswith('.pdf'):
            reader = PdfReader(self.file_path)
            for i, pdf_page in enumerate(reader.pages):
                page = Page(pdf_page.extract_text(), i+1, self.title, self.file_path)
                self.pages.append(page)
        else:
            raise ValueError("Unsupported file format. Only PDFs are supported right now.")

    def _clean_pages(self) -> None:
        """
        Cleans and processes the text of each page to remove common artifact that occur 
        on every page such as page numbers, headers, footers...etc.
        """
        tokenized_pages = [page.raw_text.split('\n') for page in self.pages]

        # Create a random sample of page numbers
        sampled_page_numbers = random.sample(range(len(tokenized_pages)), int(0.2 * len(tokenized_pages)))
        sampled_page_numbers = sorted(sampled_page_numbers)

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
                        if SequenceMatcher(None, line1, line2).ratio() > 0.9:
                            if line1 in common_lines:
                                common_lines[line1] += 1
                            elif line2 in common_lines:
                                common_lines[line2] += 1
                            else:
                                common_lines[line1] = 1
        
        # TODO - think about the threshold
        common_lines = {k for k,v in common_lines.items() if v >= (0.6 * (0.5*len(sampled_page_numbers)*(len(sampled_page_numbers)+1))) and len(k)>5}
        
        # Remove lines from page_chunks that are similar to common_lines
        for page in self.pages:
            new_page = []
            for line in page.raw_text.split('\n'):
                
                similarity = any(SequenceMatcher(None, line.strip(), common_line.strip()).ratio() > 0.90 for common_line in common_lines)
                subset = any(line.strip() in common_line.strip() or common_line.strip() in line.strip() for common_line in common_lines)
                overlap = any(overlap_score(line.strip(), common_line.strip()) > 0.80 for common_line in common_lines)
                
                if not (similarity or subset or overlap):
                    new_page.append(line)
                    
            new_page = '\n'.join(new_page)
            page.cleaned_text = new_page
    

class DocStore:
    """
    A class representing a document store that allows for the storage, retrieval, and searching of PDF documents. 
    The DocStore supports adding documents, removing them, and performing hybrid search queries using BM25 and semantic embeddings.

    Attributes:
    store (dict): A dictionary storing page embeddings and their corresponding indices in the page_store.
    title_index (defaultdict): A dictionary mapping document titles to their page indices in the page_store.
    page_store (List[Page]): A list storing Page objects compactly.
    path (str): The file path or S3 path where the store is saved.
    semantic_weight (float): The weight assigned to semantic search results in the hybrid search.
    bm25_weight (float): The weight assigned to BM25 search results in the hybrid search.
    njobs (int): The number of parallel jobs for processing.
    model (SentenceTransformer): The SentenceTransformer model used for generating semantic embeddings.
    
    Methods:
    add_document(document: Document): Adds a document to the store, finalizing its pages and saving it.
    remove_document(document_title: str): Removes a document and its pages from the store.
    list_documents(): Returns a list of document titles stored in the database.
    hybrid_search(query, top_k: int): Performs a hybrid search using BM25 and semantic search.
    bm25_search(query: str, top_k: int): Performs a BM25 search and returns the top-k matching pages.
    semantic_search(query: str, top_k: int): Performs a semantic vector search and returns the top-k matching pages.
    save(): Saves the document store to a file or S3.
    _compact_page_store(): Removes None entries from the page_store to keep it compact.
    _load(file_path: str): Loads the document store from a file or S3.
    """

    def __init__(self, path: str = None, semantic_weight: float = 0.75, bm25_weight: float = 0.25, njobs: int = None) -> None:
        self.store = {}  # Stores only embeddings and their indices in page_store
        self.title_index = defaultdict(list)  # Maps document titles to indices in page_store
        self.page_store = []  # Compact store for page objects
        self.path = path
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.njobs = njobs
        self.model = SentenceTransformer(
            "dunzhang/stella_en_400M_v5",
            trust_remote_code=True,
            device="cpu",
            config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
        )


        if path is not None:
            parsed_url = urlparse(path)
            if parsed_url.scheme == 's3':
                s3 = boto3.client('s3')
                bucket = parsed_url.netloc
                key = parsed_url.path.lstrip('/')
                try:
                    s3.head_object(Bucket=bucket, Key=key)
                    self.load(path)
                except s3.exceptions.ClientError:
                    # Create the S3 path if it doesn't exist
                    s3.put_object(Bucket=bucket, Key=key)
            else:
                if os.path.exists(path):
                    self._load(path)
                else:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, 'wb') as f:
                        pickle.dump((self.store, self.title_index, self.page_store), f)

    def add_document(self, document: Document) -> None:
        """
        Adds a document to the database, storing its pages in a compact format.
        
        Args:
        document (Document): The document to add to the database.
        """
        print(f'START FINALIZE document: {document.title}')
        document.finalize(self.njobs)
        print(f'END FINALIZE document: {document.title}')
        for page in document.pages:
            embedding = tuple(self.model.encode(page.embed_text))
            page_index = len(self.page_store)
            self.page_store.append(page)
            self.store[embedding] = page_index
            self.title_index[document.title].append(page_index)
        
        self.save()

    def remove_document(self, document_title: str) -> None:
        """
        Completely removes a document and its pages from the database.
        
        Args:
        document_title (str): The title of the document to remove.
        """
        page_indices = self.title_index.pop(document_title, [])
        for page_index in page_indices:
            # Find and remove the embedding key corresponding to this page
            embedding_to_remove = [k for k, v in self.store.items() if v == page_index]
            for emb in embedding_to_remove:
                self.store.pop(emb, None)

            # Set the page at this index to None to preserve indices of other pages
            self.page_store[page_index] = None

        # Optional: Compact the page_store if needed
        # This is a clean-up operation to remove None entries if they become too many
        self._compact_page_store()

    def list_documents(self) -> List[str]:
        """
        Lists all documents in the database.
        
        Returns:
        List[str]: A list of document titles stored in the database.
        """
        return list(self.title_index.keys())
    
    def hybrid_search(self, query, top_k=5):
        """
        Perform a hybrid search using BM25 and semantic search, and return the top results.
        
        Args:
        query (str): The search query.
        top_k (int): The number of top results to return.

        Returns:
        List[Page]: A list of top-k pages matching the search query.
        """
        vector_results = [((top_k*10)-rank,txt) for rank,txt in enumerate(self.semantic_search(query, top_k*10))]
        bm25_results = [((top_k*10)-rank,txt) for rank,txt in enumerate(self.bm25_search(query, top_k*10))]

        # Create a dictionary to store combined scores
        combined_scores = {}

        # Add BM25 results to the combined scores
        for score, doc in bm25_results:
            if doc in combined_scores:
                combined_scores[doc] += score * self.bm25_weight
            else:
                combined_scores[doc] = score * self.bm25_weight

        # Add vector search results to the combined scores
        for score, doc in vector_results:
            if doc in combined_scores:
                combined_scores[doc] += score * self.semantic_weight
            else:
                combined_scores[doc] = score * self.semantic_weight

        # Sort the combined results based on the combined scores
        combined_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # Extract the top_k results
        return [self.page_store[c[0]] for c in combined_results[:top_k]]

    def bm25_search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Perform a BM25 search over the stored documents' pages and return the top results.
        
        Args:
        query (str): The search query.
        top_k (int): The number of top results to return.
        
        Returns:
        List[int]: The top-k page indices from the BM25 search.
        """
        valid_pages = [page for page in self.page_store if page is not None]
        tokenized_corpus = [page.embed_text.split(" ") for page in valid_pages]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = query.split(" ")
        scores = bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices

    def semantic_search(self, query: str, top_k: int = 10) -> List[dict]:
        """
        Perform a semantic vector-based search over the stored documents' pages and return the top results.
        
        Args:
        query (str): The search query.
        top_k (int): The number of top results to return.
        
        Returns:
        List[int]: The top-k page indices from the BM25 search.
        """
        query_emb = self.model.encode(query, prompt_name='s2p_query')
        doc_embeddings = [np.array(k).flatten() for k in self.store.keys() if self.page_store[self.store[k]] is not None]
        similarities = self.model.similarity(query_emb, doc_embeddings)
        top_indices = np.argsort(list(similarities.flatten()))[::-1][:top_k]
        return top_indices
        
    def save(self) -> None:
        """
        Saves the database to a file, including the page_store for BM25.
        Supports saving to local file system or S3.
        """
        parsed_url = urlparse(self.path)
        if parsed_url.scheme == 's3':
            s3 = boto3.client('s3')
            bucket = parsed_url.netloc
            key = parsed_url.path.lstrip('/')
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_name = temp_file.name
                pickle.dump((self.store, self.title_index, self.page_store), temp_file)
            with open(temp_file_name, 'rb') as f:
                s3.upload_fileobj(f, bucket, key)
            os.remove(temp_file_name)
        else:
            with open(self.path, 'wb') as f:
                pickle.dump((self.store, self.title_index, self.page_store), f)

    def _compact_page_store(self):
        """
        Removes None entries from page_store to keep it compact.
        """
        new_page_store = []
        new_index_map = {}
        for i, page in enumerate(self.page_store):
            if page is not None:
                new_index_map[i] = len(new_page_store)
                new_page_store.append(page)
        
        # Update indices in store
        for k, v in list(self.store.items()):
            if v in new_index_map:
                self.store[k] = new_index_map[v]
            else:
                del self.store[k]

        self.page_store = new_page_store


    def _load(self, file_path: str) -> None:
        """
        Loads the database from a file. Supports loading from local file system or S3.
        
        Args:
        file_path (str): The path to the database file to load.
        """
        parsed_url = urlparse(file_path)
        if parsed_url.scheme == 's3':
            s3 = boto3.client('s3')
            bucket = parsed_url.netloc
            key = parsed_url.path.lstrip('/')
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_name = temp_file.name
                s3.download_fileobj(bucket, key, temp_file)
            with open(temp_file_name, 'rb') as f:
                self.store, self.title_index, self.page_store = pickle.load(f)
            os.remove(temp_file_name)
        else:
            with open(file_path, 'rb') as f:
                self.store, self.title_index, self.page_store = pickle.load(f)

def worker_finalize_page(page) -> Page:
    """
    Finalizes a single page by calling the page's finalize method and handling any exceptions.
    
    Args:
    page: A Page object to finalize.

    Returns:
    The finalized page, or None if an error occurs.
    """
    try:
        return page.finalize()
    except Exception as e:
        print(f"Error processing page: {e}")
        return None

def longest_common_substring(s1, s2):
    """
    Find the longest common substring between two strings using dynamic programming.
    
    Args:
    s1: The first string.
    s2: The second string.
    
    Returns:
    int: The length of the longest common substring.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
    
    return max_length

def overlap_score(str1: str, str2: str) -> float:
    """
    Calculate the score of the longest common substring between two strings relative to the length of the shorter string.
    
    Args:
    str1 (str): The first string to compare.
    str2 (str): The second string to compare.
    
    Returns:
    float: The overlap score, which is the length of the longest common substring divided by the length of the shorter string.
    """
    if not str1 or not str2:
        return 0.0
        
    longest_overlap = longest_common_substring(str1, str2)
    min_length = min(len(str1), len(str2))
    
    return longest_overlap / min_length
