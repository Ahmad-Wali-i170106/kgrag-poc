import base64
from enum import Enum
from typing import Union, List

import numpy as np
from PIL import Image, ImageFilter

import pytesseract

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import SystemMessage, HumanMessage

from kgrag.prompts import OCR_SYSTEM

class OCREngine(Enum):
    RAPIDOCR = "RapidOCR"
    PYTESSERACT = "pytesseract"
    LLM = "LLM"

def ocr_images_pytesseract(images: List[Union[np.ndarray | Image.Image]]) -> str:
    all_text: str = ""
    for image in images:
        if isinstance(image, Image.Image):
            # image = image.filter(ImageFilter.BLUR)
            # image = image.filter(ImageFilter.MinFilter(3))
            image = image.filter(ImageFilter.SMOOTH())
        all_text += '\n' +  pytesseract.image_to_string(image, lang="eng", config='--psm 3 --dpi 300 --oem 1')
    return all_text.strip('\n \t')

def ocr_images_llm(images: List[bytes], llm: BaseLanguageModel) -> str:
    all_text = []
    for img in images:
        image_data = base64.b64encode(img).decode("utf-8")
        messages = [
            SystemMessage(
                content=OCR_SYSTEM
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Extract text from this page. It must be legible. You will be deducted marks otherwise."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            )
        ]
        output = llm.invoke(messages).content
        all_text.append(output)
    return "\n".join(all_text)


def get_ocr_chain(llm: BaseLanguageModel) -> Runnable:
    messages = [
        SystemMessage(
            content=OCR_SYSTEM
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract text from this page. It must be legible. You will be deducted marks otherwise."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image_data}"}}
            ]
        )
    ]
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(messages)
    return prompt | llm

# def ocr_images_llm(images: List[bytes], llm: BaseLanguageModel) -> str:
    
