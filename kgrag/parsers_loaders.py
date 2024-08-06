import csv
import os
import re
from typing import Iterator, List, Optional, Union

from google.cloud import documentai
from google.cloud.documentai_toolbox import document, wrappers
from langchain_community.document_loaders.base import (
    BaseBlobParser,
    BaseLoader
)
# from langchain.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.blob_loaders import Blob
# from langchain.document_loaders.pdf import BasePDFLoader
from langchain_community.document_loaders.pdf import BasePDFLoader
# from langchain.document_loaders.base import BaseBlobParser, BaseLoader
# from langchain.schema import Document
from langchain_core.documents import Document

from kgrag.fileio import delete_folder, file_exists


class GoogleCloudDocumentAIOnlineParser(BaseBlobParser):
    """Load `PDF' using Document AI and chunk at character level"""

    def __init__(self, **kwargs):
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai

        project_id = os.environ.get("PROJECT_ID","onediamond-staging") # "cedar-gift-317820"
        location = os.environ.get("PROCESSOR_LOCATION","us")  # Format is 'us' or 'eu'
        processor_id = os.environ.get("DOCAI_PROCESSOR_ID","4d39007f411e8657")  # Create processor in Cloud Console "b0bd3522166af1c5"
        
        self.mime_type = kwargs.get("mime_type","application/pdf")
        self.docai_client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        )
        self.resource_name = self.docai_client.processor_path(project_id, location, processor_id)
    
    def lazy_parse(self, blob: Blob, uid=-1) -> Iterator[Document]:
        from google.cloud import documentai

        pdf_file_obj = blob.as_bytes()
        
        # Load Binary Data into Document AI RawDocument Object
        raw_document = documentai.RawDocument(content=pdf_file_obj, mime_type=self.mime_type)

        ocr_config = documentai.OcrConfig(enable_native_pdf_parsing=True) # {"ocr_config": {"enable_native_pdf_parsing": True}}
        
        project_name = self.docai_client.common_project_path()
        # print(project_name)

        # Configure the process request
        request = documentai.ProcessRequest(
            name=self.resource_name, 
            raw_document=raw_document, 
            skip_human_review=True, 
            process_options=documentai.ProcessOptions(ocr_config=ocr_config),
        )

        # Use the Document AI client to process the sample form
        result = self.docai_client.process_document(request=request)

        document_object = result.document
        pages = document_object.pages
        
        
        meta_content = f"{blob.source} has {len(pages)} pages, {len(document_object.text.split(' '))+1} words and {len(document_object.text)} characters."
        
        try:
            if document_object.uri is not None:
                meta_content += f"\n{blob.source} has the GCP Bucket URI: {document_object.uri}"
        except:
            pass
        
        try:
            if document_object.entities is not None:
                entities = [f"{ent.mention_text} with id {ent.id} of type {ent.type_}" for ent in document_object.entities]
                entities = '\n'.join(entities)
                meta_content += f"\n{blob.source} contains the following entities:\n{entities}."
        except:
            pass

        meta_doc = Document(
            page_content=meta_content,
            metadata={
                "filename":str(blob.source).lower(),
                "source": f"{blob.source}",
                "user_id": f"{uid}"
            }
        )

        yield from [
            Document(
                page_content=page.layout.text_anchor.content,
                metadata={
                    "filename": str(blob.source).lower(),
                    "source": f"{blob.source}: Pg. {page.page_number}",
                    "user_id": f"{uid}"
                }
            )
            for page in pages
        ] + [meta_doc]

    def parse(self, blob: Blob, uid=-1) -> List[Document]:
        return list(self.lazy_parse(blob, uid))

class GoogleCloudDocumentAIBatchParser(BaseBlobParser):
    
    """Load `PDF` from given file_uri and chunk at character level"""
    def __init__(self, **kwargs):
        from google.api_core.client_options import ClientOptions
        from google.cloud import documentai, storage

        self.output_tables_folder = kwargs.get('output_tables_folder', os.environ.get("TABLES_FOLDERPATH", "."))
        self.location = os.environ.get("PROCESSOR_LOCATION","us")  # Format is "us" or "eu"
        processor_id = os.environ.get("DOCAI_PROCESSOR_ID","4d39007f411e8657") #,"b0bd3522166af1c5") #

        self.mime_type = "application/pdf"
        # Optional. The fields to return in the Document object.
        self.field_mask = "uri,shardInfo,text,entities,pages.pageNumber,pages.layout,pages.tables"

        self.docai_client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        )
        
        self.storage_client = storage.Client()
        project_id = self.storage_client.project #os.environ.get("PROJECT_ID","onediamond-staging") # "cedar-gift-317820" #self.docai_client.project
        self.resource_name = self.docai_client.processor_path(project_id, self.location, processor_id)
        
    
    @staticmethod
    def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
        """
        Document AI identifies text in different parts of the document by their
        offsets in the entirety of the document"s text. This function converts
        offsets to a string.
        """
        # If a text segment spans several lines, it will
        # be stored in different text segments.
        return " ".join(
            text[int(segment.start_index) : int(segment.end_index)]
            for segment in layout.text_anchor.text_segments
        )

    @staticmethod
    def extract_table(table: wrappers.page.Table, output_filename: Optional[str] = None):
        df = table.to_dataframe()
        # Write Dataframe to CSV file
        if output_filename is not None and output_filename != "":
            df.to_csv(output_filename, sep='|', index=False, na_rep="NA",
                      float_format=lambda a:"{:,.2f}".format(a),
                      quoting=csv.QUOTE_NONE, escapechar='\\')
            return ""
        else:
            return df.to_csv(sep='|', index=False, na_rep="NA", float_format=lambda a:"{:,.2f}".format(a),
                        quoting=csv.QUOTE_NONE, escapechar='\\')

    def lazy_parse(self, user_id: int, gcs_input_uris: Union[List[str],str], filenames: Union[str, List[str]],
                   gcs_output_uri: Optional[str]=None, timeout: int=500) -> Iterator[Document]:
        
        from google.api_core.exceptions import InternalServerError, RetryError
        from google.cloud import documentai


        if gcs_output_uri is None:
            gcs_output_uri = os.environ.get("GCS_OUTPUT_FOLDER","gs://qna-staging/docs/outputs/")

        if isinstance(gcs_input_uris, str):
            gcs_input_uris = [gcs_input_uris]
        
        if isinstance(filenames, str):
            filenames = [filenames]
        
        gcs_documents = [documentai.GcsDocument(
            gcs_uri=gcs_input_uris[i], mime_type=self.mime_type
        ) for i in range(len(gcs_input_uris))]
        # Load GCS Input URIs into a List of document files
        gcs_documents = documentai.GcsDocuments(documents=gcs_documents)
        input_config = documentai.BatchDocumentsInputConfig(gcs_documents=gcs_documents)
        
        gcs_output_config = documentai.DocumentOutputConfig.GcsOutputConfig(
            gcs_uri=gcs_output_uri, field_mask=self.field_mask
        )

        # Where to write results
        output_config = documentai.DocumentOutputConfig(gcs_output_config=gcs_output_config)
        
        ocr_config = documentai.OcrConfig(enable_native_pdf_parsing=True)

        request = documentai.BatchProcessRequest(
            name=self.resource_name,
            input_documents=input_config,
            document_output_config=output_config,
            process_options=documentai.ProcessOptions(ocr_config=ocr_config)
        )
        # BatchProcess returns a Long Running Operation (LRO)
        operation = self.docai_client.batch_process_documents(request)

        # Continually polls the operation until it is complete.
        # This could take some time for larger files
        # Format: projects/{project_id}/locations/{location}/operations/{operation_id}
        try:
            print(f"Waiting for operation {operation.operation.name} to complete...")
            operation.result(timeout=timeout)
        # Catch exception when operation doesn"t finish before timeout
        except (RetryError, InternalServerError) as e:
            print(e)

        metadata = documentai.BatchProcessMetadata(operation.metadata)

        if metadata.state != documentai.BatchProcessMetadata.State.SUCCEEDED:
            print(f"Batch Process Failed: {metadata.state_message}")
            raise ValueError(f"Batch Process Failed: {metadata.state_message}")
        print(f"Batch operation succeeded!")
        wrapped_document = document.Document.from_batch_process_metadata(metadata)
        # wrapped_document = document.Document.from_batch_process_operation(location=self.location, operation_name=operation.operation.name)
        all_documents = []
        # print(wrapped_document)
        for i, doc in enumerate(wrapped_document):
            filename = filenames[i] or doc.gcs_input_uri.split('/')[-1]
            page_docs = []
            print(f"Fetching and creating docs from {doc.gcs_prefix} for {filename}...")
            for page in doc.pages:
                print(f"Processing Page {page.page_number}/{len(doc.pages)}...")
                page_docs.append(
                    Document(
                        page_content = page.text,
                        metadata = {
                            "page": page.page_number,
                            "source": f"{filename}:  Pg. {page.page_number}",
                            "filename": filename,
                            "user_id": f"{user_id}"
                        }
                    )
                )
                tables_list = []
                for ti, table in enumerate(page.tables):
                    # Save each extracted table to a predefined folder
                    output_table_filename = os.path.join(self.output_tables_folder,f"{filename}-{page.page_number}-{ti}.csv")
                    # print(output_table_filename)
                    _ = self.extract_table(table, output_table_filename)
                    tables_list.append(output_table_filename)
                # print(tables_list)
                if len(tables_list) > 0:
                    page_docs.append(
                        Document(
                            page_content = ','.join(tables_list),
                            metadata = {
                                "page": page.page_number,
                                "source": f"{filename}:  Pg. {page.page_number}",
                                "filename": filename,
                                "user_id": f"{user_id}",
                                "start_index": -2
                            }
                        )
                    )
            
            npages = len(doc.pages)
            nwords = len(doc.text.split(' '))
            nchars = len(doc.text)
            meta_content = f"{filename} has {npages} pages, {nwords} words and {nchars} characters."
            meta_doc = Document(
                page_content=meta_content,
                metadata = {
                    "page": 0,
                    "source": filename,
                    "filename": filename,
                    "user_id": f"{user_id}",
                    "start_index": 0
                }
            )
            
            all_documents.extend(page_docs+[meta_doc])
            delete_folder(folder_path=doc.gcs_prefix, client=self.storage_client, bucket_name=doc.gcs_bucket_name)
        
        yield from all_documents

    def parse(self, user_id: int, gcs_input_uris: Union[List[str],str], filenames: Union[str, List[str]],
              gcs_output_uri: Optional[str]=None, timeout: int=300) -> List[Document]:
        return list(self.lazy_parse(user_id, gcs_input_uris, filenames, gcs_output_uri, timeout=timeout))
        


class PyPDF2Parser(BaseBlobParser):
    """Load `PDF` using `PyPDF2` and chunk at character level."""
    def __init__(self, password: Optional[Union[str, bytes]] = None):
        self.password = password

    def lazy_parse(self, blob: Blob, uid=-1) -> Iterator[Document]:
        import PyPDF2

        """Lazily parse the blob"""
        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = PyPDF2.PdfReader(pdf_file_obj, password=self.password)
            meta = pdf_reader.metadata
            
            
            pages_docs = [
                Document(
                    page_content=page.extract_text(),
                    metadata={
                        "filename":str(blob.source).split('/')[-1].lower(), 
                        "page":page_number+1, 
                        "source":f"{blob.source.split('/')[-1]}: Pg. {page_number+1}",
                        "user_id": f"{uid}"
                        }
                )
                for page_number, page in enumerate(pdf_reader.pages)
            ]
            
            total_words = sum([len(re.findall(r"\w+", pdoc.page_content)) for pdoc in pages_docs])
            total_chars = sum([len(pdoc.page_content) for pdoc in pages_docs])
            meta_content = f"{blob.source} has {len(pdf_reader.pages)} pages, {total_words} words, and {total_chars} characters."
            
            if meta.title is not None:
                meta_content += f"\n{blob.source} has the title {meta.title}."
            if meta.author is not None:
                meta_content += f"\n{blob.source} has the author {meta.author}."
            if meta.subject is not None:
                meta_content += f"\n{blob.source} has subject: {meta.subject}."
            
            meta_doc = Document(
                page_content=meta_content,
                metadata={
                    "filename":str(blob.source).split('/')[-1].lower(),
                    "page":page_number+1, 
                    "source": f"{blob.source.split('/')[-1]}",
                    "user_id": f"{uid}"
                }
            )

            yield from pages_docs + [meta_doc]

    def parse(self, blob: Blob, uid=-1) -> List[Document]:
        return list(self.lazy_parse(blob, uid))
    
class TesseractOCRParser(BaseBlobParser):
    """Load `PDF` using 'PyTesseract` OCR and chunk at character level"""

    def lazy_parse(self, blob: Blob, uid=-1) -> Iterator[Document]:
        
        import pdf2image
        import pytesseract

        pdf_bytes = blob.as_bytes()
        images = pdf2image.convert_from_bytes(pdf_bytes)

        pages_docs = [
            Document(
                page_content=pytesseract.image_to_string(page, config="--psm 3"),
                metadata={
                    "filename":str(blob.source).lower(), 
                    #"page":page_number+1, 
                    "source":f"{blob.source}: Pg. {page_number+1}",
                    "user_id": f"{uid}"
                    }
            )
            for page_number, page in enumerate(images)
        ]

        total_words = sum([len(pdoc.page_content.split(' '))+1 for pdoc in pages_docs])
        total_chars = sum([len(pdoc.page_content) for pdoc in pages_docs])
        meta_content = f"{blob.source} has {len(images)} pages, {total_words} words and {total_chars} characters."
        
        meta_doc = Document(
            page_content=meta_content,
            metadata={
                "filename":str(blob.source).lower(),
                "source": f"{blob.source}",
                "user_id": f"{uid}"
            }
        )

        yield from pages_docs + [meta_doc]

    def parse(self, blob: Blob, uid=-1) -> List[Document]:
        return list(self.lazy_parse(blob, uid))

class DocAI_Loader(BaseLoader):
    """Load `PDF` file from a given google cloud storage URI"""

    def __init__(self, file_uris: Union[str, List[str]], filenames: Union[str, List[str]], output_folder_uri: Optional[str]=None, user_id=None, **kwargs):
        if user_id is None:
            self.user_id = -1
        else:
            self.user_id = user_id

        self.parser = GoogleCloudDocumentAIBatchParser(**kwargs)
        self.filenames = filenames
        self.file_uris = [DocAI_Loader.__validate_uri(file_uri) for file_uri in file_uris]
        self.file_uris = [f for f in self.file_uris if len(f) > 0]
        if output_folder_uri is None:
            self.output_folder_uri = os.environ.get("GCS_OUTPUT_FOLDER","gs://qna-staging/docs/outputs/")
        else:
            self.output_folder_uri = DocAI_Loader.__validate_uri(output_folder_uri).strip('/') + '/'
    
    @staticmethod
    def __validate_uri(uri:str):
        uri = uri.strip('/')
        if not uri.startswith("gs"):
            uri = "gs://" + uri.strip(":/")
            return uri
        # return uri
        try:
            if file_exists(uri):
                return uri
        except:
            return ""
        return ""

    def load(self) -> List[Document]:
        return self.parser.parse(self.user_id, gcs_input_uris=self.file_uris, filenames=self.filenames,
                                     gcs_output_uri=self.output_folder_uri)

    def lazy_load(self) -> Iterator[Document]:
        yield from self.parser.parse(self.user_id, gcs_input_uris=self.file_uris, filenames=self.filenames,
                                     gcs_output_uri=self.output_folder_uri)   

class PyPDF2_TesseractOCR_Loader(BasePDFLoader):
    """Load `PDF` using either TesseractOCR or PyPDF2

    Loader also stores page numbers in data
    """
    def __init__(self, file_path: str, filename: Optional[str] = None, data: Optional[Union[bytes,str]] = None, 
                 password: Optional[Union[str, bytes]] = None, use_docai: bool = False,
                 use_ocr: bool = False, user_id=None) -> None:
        
        if user_id is None:
            self.user_id = -1
        else:
            self.user_id = user_id
        
        if use_ocr:
            if use_docai:
                self.parser = GoogleCloudDocumentAIOnlineParser()
            else:
                self.parser = TesseractOCRParser()
        else:
            self.parser = PyPDF2Parser(password=password)
        self.data = data
        if file_path is None:
            self.file_path = filename
        else:
            super().__init__(file_path) #, data
    
    def load(self) -> List[Document]:
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        blob = None
        if self.data is not None:
            blob = Blob.from_data(self.data, path=self.file_path)
        elif self.file_path is not None:
            blob = Blob.from_path(self.file_path)
        else:
            raise Exception("Need either file_path or data or both. Both can not be None.")
        yield from self.parser.parse(blob, uid=self.user_id)
