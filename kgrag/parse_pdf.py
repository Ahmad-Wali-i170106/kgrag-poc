
from functools import partial
from typing import Any, Callable, Iterable, List, Dict, Optional
import pymupdf

import numpy as np
from PIL import Image
from langchain_core.language_models import BaseLanguageModel


from kgrag.data_schema_utils import extract_chapter

EXCLUDED_KEYS = ["modDate", "creationDate"]

from kgrag.ocr import OCREngine


class PDFParser():
    def __init__(
        self, 
        pdf_path: str, 
        ocr_engine: str | OCREngine = OCREngine.PYTESSERACT,
        llm: Optional[BaseLanguageModel] = None, 
        ocr_llm: Optional[BaseLanguageModel] = None
    ) -> None:
        self.doc = pymupdf.Document(pdf_path)
        self.llm: BaseLanguageModel | None = llm
        if llm:
            self.use_llm = True
        else:
            self.use_llm = False
        self.filepath: str = pdf_path.split('/')[-1]
        if isinstance(ocr_engine, str):
            try:
                ocr_engine = OCREngine[ocr_engine.upper()]
            except KeyError:
                print(f"{ocr_engine} is not a valid OCREngine. Valid values are {OCREngine._member_names_}",
                      f"Assuming the default value: {OCREngine.PYTESSERACT}")
                ocr_engine = OCREngine.PYTESSERACT
        self.ocr_engine = ocr_engine
        self._ocr_func: Callable
        if self.ocr_engine == OCREngine.LLM:
            if ocr_llm is None or not isinstance(ocr_llm, BaseLanguageModel):
                raise ValueError("You must provide ocr_llm of type `langchain.llm.base.BaseLanguageModel` if you've chosen a multimodel LLM to perform OCR",
                                 "Otherwise, try using another OCR engine instead: PYTESSERACT | RAPIDOCR")
            
            from kgrag.ocr import ocr_images_llm
            
            self._ocr_func = partial(ocr_images_llm(llm=ocr_llm))
        elif self.ocr_engine == OCREngine.PYTESSERACT:
            try:
                import pytesseract
            except ImportError:
                raise Exception("You do not have pytesseract installed",
                                "Install it using `pip install pytesseract`",
                                "Or try using another OCR engine instead: LLM | RAPIDOCR")
            
            from kgrag.ocr import ocr_images_pytesseract

            self._ocr_func = ocr_images_pytesseract
        elif self.ocr_engine == OCREngine.RAPIDOCR:
            try:
                import rapidocr_onnxruntime
            except ImportError:
                raise Exception("You do not have rapidocr_onnxruntime installed",
                                "Install it using `pip install rapidocr_onnxruntime`",
                                "Or try using another OCR engine instead: LLM | PYTESSERACT")
            from langchain_community.document_loaders.parsers.pdf import extract_from_images_with_rapidocr
            
            self._ocr_func = extract_from_images_with_rapidocr

    def get_toc(self) -> List[List[int]]:
        return self.doc.get_toc()
        # return self.toc

    def extract_page_metadata(self, page_num: int, page_text: str) -> Dict[str, Any]:
        md = extract_chapter(text=page_text, model=self.llm)
        # md["page_num"] = page_num
        if md.get("chapter_title", None) is not None:
            if len(self.pdf_data) > 0:
                md["chapter_title"] = self.pdf_data[-1]["page_metadata"]["chapter_title"]
                md["chapter_number"] = self.pdf_data[-1]["page_metadata"]["chapter_num"]
        return md

    def get_page_metadata(self, page_num: int, page_text: str, toc: list) -> dict[str, Any] | dict[str, int] | Any | Any:
        if toc:
            # find page metadata using the table of contents...
            chapter_title: str | None = None
            chapter_number: int = 1
            chapter_start_at_one: bool = True
            for i in range(len(toc)):
                try:
                    if isinstance(toc[i][0], float): 
                        if toc[i][0] != int(toc[i][0]): 
                            chapter_start_at_one = False
                            continue
                    elif isinstance(toc[i][0], str): 
                        if '.' in toc[i][0] or ',' in toc[i][0]:
                            chapter_start_at_one = False
                            continue
                    if chapter_start_at_one and int(toc[i][0]) != 1:
                        continue
                    if page_num >= toc[i][2]: #and ("chapter" in toc[i][1] or "Chapter" in toc[i][1] or "CHAPTER" in toc[i][1]):
                        chapter_title = toc[i][1]
                        chapter_number = i+1
                    else:
                        i = len(toc)
                except Exception as e:
                    print(e)
                    continue
            page_metadata = {
                "page": page_num,
                "source": f"{self.filepath} Page: {page_num}",
                "chapter_title": chapter_title,
                "chapter_number": chapter_number
            }
        elif not self.use_llm:
            page_metadata = {
                "page": page_num,
                "source": f"{self.filepath} Page: {page_num}"
            }
        else:
            if len(page_text.strip()) <= 5:
                # if the text is too small, there's no point trying to look for a chapter within 
                page_metadata = {}
            else:
                page_metadata = self.extract_page_metadata(page_num, page_text)
            page_metadata["page"] = page_num
            page_metadata["source"] = f"{self.filepath} Page: {page_num}"
        return page_metadata



    def extract_page_info(self, page_num: int, page: pymupdf.Page, doc_metadata: dict, toc: list = None):
        """Extracts text and metadata from a page of a PDF document.

        Args:
            page_num: int
            page: pymupdf.Page
            doc_metadata: dict
        """
        page_info = {}

        text_content: str = page.get_text()
        images_list: list[list] = page.get_images()
        if len(images_list) > 0:
            imgs = []
            for img in images_list:
                xref = img[0]
                pix = pymupdf.Pixmap(self.doc, xref)
                if self.ocr_engine == OCREngine.PYTESSERACT:
                    cspace = pix.colorspace
                    if cspace is None:
                        mode: str = "L"
                    elif cspace.n == 1:
                        mode = "L" if pix.alpha == 0 else "LA"
                    elif cspace.n == 3:
                        mode = "RGB" if pix.alpha == 0 else "RGBA"
                    else:
                        mode = "CMYK"
                    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                    if mode != "L":
                        img = img.convert("L")
                    imgs.append(img)
                elif self.ocr_engine == OCREngine.RAPIDOCR:
                    imgs.append(
                        np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.height, pix.width, -1
                        )
                    )
                elif self.ocr_engine == OCREngine.LLM:
                    base_image = self.doc.extract_image(xref)  # Extract the image
                    image_bytes = base_image['image']  # Get the image bytes
                    
                    imgs.append(image_bytes)
            text_content += '\n' + self._ocr_func(imgs)
        text_content = text_content.strip(' \n\t')
        page_info["text"] = text_content
        page_info["page_metadata"] = self.get_page_metadata(page_num=page_num, page_text=text_content, toc=toc)
        page_info["doc_metadata"] = doc_metadata
        return page_info

    def get_document_metadata(self) -> Dict[str, Any]:
        """Extracts metadata from a PDF document. """
        
        if self.doc.metadata is None:
            return {}
        
        doc_metadata = {}
        for k, v in self.doc.metadata.items():
            if k in EXCLUDED_KEYS or v is None:
                continue
            elif isinstance(v, str):
                if len(v.strip()) > 0:
                    doc_metadata[k] = v.strip()
            elif isinstance(v, Iterable) and not isinstance(v, str):
                if len(v) > 0:
                    doc_metadata[k] = v
            else:
                doc_metadata[k] = v
        doc_metadata['filename'] = self.filepath
        return doc_metadata

    def process_pdf_document(self, num_of_pages: int = None) -> List[Dict[str, Any]]:
        if not num_of_pages:
            num_of_pages = len(self.doc)
        self.pdf_data = []
        toc = self.get_toc()
        doc_metadata = self.get_document_metadata()
        for page_num, page in enumerate(self.doc):
            if page_num < num_of_pages:
                self.pdf_data.append(self.extract_page_info(page_num=page_num+1, page=page, doc_metadata=doc_metadata, toc=toc))
            else:
                break
        self.doc.close()
        return self.pdf_data


class PDFParserMarkdown:
    def __init__(
            self, 
            pdf_path: str, 
            pages: int | List[int] | None = None,
            ocr_engine: str | OCREngine = OCREngine.PYTESSERACT,
            llm: Optional[BaseLanguageModel] = None, 
            ocr_llm: Optional[BaseLanguageModel] = None
        ) -> None:
        '''
        Use PyMuPDF and PyMuPDF4LLM to read input PDF document and convert to markdown.
        If the PDF contains images, it will always OCR all the images in the PDF to extract any text.
        
        Args:
            pdf_path (str): String path to open file
            pages (List[int]| int): List of page numbers (zero-indexed) to read from the PDF.
                                    If an integer is provided, reads all pages from beginning to the provided page number.
                                    Default=All pages
            ocr_engine (str | OCREngine): Which OCR engine to use. There are 3 options: PYTESSERACT, RAPIDOCR & LLM. (LLM is most accurate but also most expensive, PYTESSERACT works for most cases with 70%+ accuracy)
            llm (BaseLanguageModel): Can optionally provide an LLM to use to deduce chapter number of each page (if the input file has no table-of-contents)
            ocr_llm (BaseLanguageModel): If ocr_engine="LLM", must provide ocr_llm (a multimodel LLM like "gpt-4o" or "gemini-1.5-flash") to perform OCR.
        '''
        import pymupdf4llm

        self.doc = pymupdf.Document(pdf_path)
        if pages is None:
            pages = list(range(0, len(self.doc)))
        if isinstance(pages, int):
            pages = list(range(0, pages))
        self.page_numbers = pages
        self.doc_markdown = pymupdf4llm.to_markdown(self.doc, pages=self.page_numbers, page_chunks=True, write_images=False, force_text=True)
        self.llm: BaseLanguageModel | None = llm
        if llm:
            self.use_llm = True
        else:
            self.use_llm = False
        self.filepath: str = pdf_path.split('/')[-1]
        if isinstance(ocr_engine, str):
            try:
                ocr_engine = OCREngine[ocr_engine.upper()]
            except KeyError:
                print(f"{ocr_engine} is not a valid OCREngine. Valid values are {OCREngine._member_names_}",
                      f"Assuming the default value: {OCREngine.PYTESSERACT}")
                ocr_engine = OCREngine.PYTESSERACT
        self.ocr_engine = ocr_engine
        if self.ocr_engine == OCREngine.LLM:
            if ocr_llm is None or not isinstance(ocr_llm, BaseLanguageModel):
                raise ValueError("You must provide ocr_llm of type `langchain.llm.base.BaseLanguageModel` if you've chosen a multimodel LLM to perform OCR")
            
            from kgrag.ocr import ocr_images_llm
            
            self._ocr_func = partial(ocr_images_llm(llm=ocr_llm))
        elif self.ocr_engine == OCREngine.PYTESSERACT:
            try:
                import pytesseract
            except ImportError:
                raise Exception("You do not have pytesseract installed",
                                "Install it using `pip install pytesseract`",
                                "Or try using another OCR engine instead: LLM | RAPIDOCR")
            
            from kgrag.ocr import ocr_images_pytesseract

            self._ocr_func = ocr_images_pytesseract
        elif self.ocr_engine == OCREngine.RAPIDOCR:
            try:
                import rapidocr_onnxruntime
            except ImportError:
                raise Exception("You do not have rapidocr_onnxruntime installed",
                                "Install it using `pip install rapidocr_onnxruntime`",
                                "Or try using another OCR engine instead: LLM | RAPIDOCR")
            from langchain_community.document_loaders.parsers.pdf import extract_from_images_with_rapidocr
            
            self._ocr_func = extract_from_images_with_rapidocr
        self.chapter_count = 0

    def get_toc(self) -> List[List[int]]:
        return self.doc.get_toc()
        # return self.toc

    def extract_page_metadata(self, page_text: str) -> Dict[str, Any]:
        page_text = '\n'.join(page_text.split('\n')[:5])
        md: Dict[str, Any] = extract_chapter(text=page_text, model=self.llm)
        # md["page_num"] = page_num
        chapter_title = md.get("chapter_title", None)
        if  chapter_title is None or len(chapter_title.strip()) < 2:
            if len(self.pdf_data) > 0:
                md["chapter_title"] = self.pdf_data[-1]["page_metadata"].get("chapter_title", "")
                md["chapter_number"] = self.pdf_data[-1]["page_metadata"].get("chapter_number", self.chapter_count)
            else:
                md["chapter_title"] = ""
                md['chapter_number'] = self.chapter_count
        else:
            self.chapter_count += 1
            md['chapter_number'] = self.chapter_count
        return md

    def get_page_metadata(self, page_num: int, page_text: str, toc: list) -> Dict[str, Any]:
        if toc:
            # find page metadata using the table of contents...
            chapter_title: str | None = None
            chapter_number: int = 1
            chapter_start_at_one: bool = True
            for i in range(len(toc)):
                try:
                    if isinstance(toc[i][0], float): 
                        if toc[i][0] != int(toc[i][0]): 
                            chapter_start_at_one = False
                            continue
                    elif isinstance(toc[i][0], str): 
                        if '.' in toc[i][0] or ',' in toc[i][0]:
                            chapter_start_at_one = False
                            continue
                    if chapter_start_at_one and int(toc[i][0]) != 1:
                        continue
                    if page_num >= toc[i][2]: #and ("chapter" in toc[i][1] or "Chapter" in toc[i][1] or "CHAPTER" in toc[i][1]):
                        chapter_title = toc[i][1]
                        chapter_number = i+1
                    else:
                        i = len(toc)
                except Exception as e:
                    print(e)
                    continue
            page_metadata = {
                "page": page_num,
                "source": f"{self.filepath} Page: {page_num}",
                "chapter_title": chapter_title,
                "chapter_number": chapter_number
            }
        elif not self.use_llm:
            page_metadata = {
                "page": page_num,
                "source": f"{self.filepath} Page: {page_num}",
                "chapter_number": self.chapter_count
            }
        else:
            if len(page_text.strip()) <= 10:
                # if the text is too small, there's no point trying to look for a chapter within 
                page_metadata = {}
            else:
                page_metadata = self.extract_page_metadata(page_text)
            page_metadata["page"] = page_num
            page_metadata["source"] = f"{self.filepath} Page: {page_num}"
        return page_metadata

    def extract_page_info(self, ip: int, page_num: int, doc_metadata: dict, toc: list = None) -> Dict[str, Any]:
        """Extracts text and metadata from a page of a PDF document.

        Args:
            page_num: int
            doc_metadata: dict
        """
        page_info = {}
        page = self.doc[page_num]
        # text_content: str = page.get_text()
        text_content: str = self.doc_markdown[ip]['text']
        images_list: list[list] = page.get_images()
        if len(images_list) > 0:
            imgs = []
            for img in images_list:
                xref = img[0]
                pix = pymupdf.Pixmap(self.doc, xref)
                if self.ocr_engine == OCREngine.PYTESSERACT:
                    cspace = pix.colorspace
                    if cspace is None:
                        mode: str = "L"
                    elif cspace.n == 1:
                        mode = "L" if pix.alpha == 0 else "LA"
                    elif cspace.n == 3:
                        mode = "RGB" if pix.alpha == 0 else "RGBA"
                    else:
                        mode = "CMYK"
                    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                    if mode != "L":
                        img = img.convert("L")
                    imgs.append(img)
                elif self.ocr_engine == OCREngine.RAPIDOCR:
                    imgs.append(
                        np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                            pix.height, pix.width, -1
                        )
                    )
                elif self.ocr_engine == OCREngine.LLM:
                    base_image = self.doc.extract_image(xref)  # Extract the image
                    image_bytes = base_image['image']  # Get the image bytes
                    
                    imgs.append(image_bytes)
            text_content += '\n' + self._ocr_func(imgs)
            # if self.ocr_engine == OCREngine.PYTESSERACT:   
            #     text_content += '\n' + ocr_images_pytesseract(imgs)
            # elif self.ocr_engine == OCREngine.RAPIDOCR:
            #     text_content += '\n' + #extract_from_images_with_rapidocr(imgs)
            # elif self.ocr_engine == OCREngine.LLM:
            #     # ocr_output = self.ocr_chain.batch([{"image_data": img} for img in imgs])
            #     text_content += '\n' + ocr_images_llm(imgs, self.ocr_llm)
        text_content = text_content.strip(' \n\t')
        page_info["text"] = text_content
        page_info["page_metadata"] = self.get_page_metadata(page_num=page_num+1, page_text=text_content, toc=toc)
        page_info["doc_metadata"] = doc_metadata
        return page_info

    def get_document_metadata(self) -> Dict[str, Any]:
        """Extracts metadata from a PDF document. """
        
        if self.doc.metadata is None:
            return {}
        
        doc_metadata = {}
        for k, v in self.doc.metadata.items():
            if k in EXCLUDED_KEYS or v is None:
                continue
            elif isinstance(v, str):
                if len(v.strip()) > 0:
                    doc_metadata[k] = v.strip()
            elif isinstance(v, Iterable) and not isinstance(v, str):
                if len(v) > 0:
                    doc_metadata[k] = v
            else:
                doc_metadata[k] = v
        doc_metadata['filename'] = self.filepath
        return doc_metadata

    def process_pdf_document(self) -> List[Dict[str, Any]]:
        self.chapter_count = 0
        self.pdf_data = []
        toc = self.get_toc()
        doc_metadata = self.get_document_metadata()
        for i, page_num in enumerate(self.page_numbers):
            self.pdf_data.append(self.extract_page_info(ip=i, page_num=page_num, doc_metadata=doc_metadata, toc=toc))
            
        self.doc.close()
        return self.pdf_data


