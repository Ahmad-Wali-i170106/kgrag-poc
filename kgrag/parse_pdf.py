from typing import Any, Literal, List, Dict
from kgrag.prompt import extract_chapter
import os
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata"
import pymupdf

class PDFParser():
  def __init__(self, pdf_path: str, llm = None):
    self.doc = pymupdf.open(pdf_path)
    self.llm = llm
    if llm:
      self.use_llm = True
    else:
      self.use_llm = False
  
  def get_toc(self):
    self.toc = self.doc.get_toc()
    return self.toc
  
  def extract_page_metadata(self, page_num: int, page_text: str):
    md = extract_chapter(text=page_text, model=self.llm)
    md["page_num"] = page_num
    if not md["chapter_title"]:
      if len(self.pdf_data) > 0:
        md["chapter_title"] = self.pdf_data[-1]["page_metadata"]["chapter_title"]
        md["chapter_num"] = self.pdf_data[-1]["page_metadata"]["chapter_num"]
    return md
  
  def get_page_metadate(self, page_num: int, page_text: str, toc: list):
    if not self.use_llm:
      if toc:
        # find page metadata using the table of contents...
        chapter_title = None
        for i in range(len(toc)):
          # print(toc[i][2])
          if page_num >= toc[i][2] and ("chapter" in toc[i][1] or "Chapter" in toc[i][1] or "CHAPTER" in toc[i][1]):
            chapter_title = toc[i][1]
          
        page_metadata = {
          "page_num": page_num,
          "chapter_title": chapter_title
        }
      else:
        page_metadata = {
          "page_num": page_num
        }
    else:
      page_metadata = self.extract_page_metadata(page_num, page_text)
      page_metadata["page_num"] = page_num
    return page_metadata
  


  def extract_page_info(self, page_num: int, page, doc_metadata: dict, toc: list = None):
    """Extracts text and metadata from a page of a PDF document.

    Args:
      page_num: int
      page: pymupdf.Page
      doc_metadata: dict
    """
    text_content = ""
    page_info = {}
    text = page.get_text()
    text_content += text
    images_list = page.get_images()
    if images_list:
      for img in images_list:
        xref = img[0]
        pix = pymupdf.Pixmap(self.doc, xref) # create a Pixmap
        if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
          pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        pdf_img = pymupdf.open("pdf", pix.pdfocr_tobytes())
        for pg in pdf_img:
          img_text = pg.get_text()
          text_content += img_text
        pdf_img.close()
    page_info["text"] = text_content
    page_info["page_metadata"] = self.get_page_metadate(page_num=page_num, page_text=text_content, toc=toc)
    page_info["doc_metadata"] = doc_metadata
    return page_info
    


  def get_document_metadata(self) -> Dict[str, Any | Literal['']] | None:
    """Extracts metadata from a PDF document. """
    return self.doc.metadata
  
  def process_pdf_document(self, num_of_pages: int = None) -> List[Dict[str, Any]]:
    if not num_of_pages:
      num_of_pages = len(self.doc)
    self.pdf_data = []
    toc = self.get_toc()
    doc_metadata = self.get_document_metadata()
    for page_num, page in enumerate(self.doc):
      if page_num <= num_of_pages:
        self.pdf_data.append(self.extract_page_info(page_num=page_num+1, page=page, doc_metadata=doc_metadata, toc=toc))
      else:
        return self.pdf_data
    self.doc.close()
    return self.pdf_data
  
  # def load_as_documents(self, num_of_pages: int = None) -> list:
  #   docs = self.process_pdf_document()
  #   return [
  #     Document
  #   ]


"""
1) Extract metadata from the pdf document
2) Read page of the pdf.
3) Extract text from page
4) if images in page: extract image.
5) Extract text from image. 
6) combine text from images to text extracted from the page.
7) if metadata in page: extract metadata
8) for a single pdf, create a list of dictionaries with keys: {"text", "page_metadata", "doc_metadata"}. List of dictionaries: [{"text", "page_metadata", "doc_metadata"}, {"text", "page_metadata", "doc_metadata"}, {"text", "page_metadata", "doc_metadata"}, ..]

"""
