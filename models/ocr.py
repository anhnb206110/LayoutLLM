from transformers import LayoutLMv3Processor, LayoutLMv3Model
from PIL import Image
import numpy as np

def load_LayoutLMv3(model_name='microsoft/layoutlmv3-base', apply_ocr=False):
    layoutlm_processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=apply_ocr)
    layoutlm_model = LayoutLMv3Model.from_pretrained(model_name)
    return layoutlm_processor, layoutlm_model

def load_PaddleOCR():
    from paddleocr import PaddleOCR, draw_ocr
    # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
    # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True,
                    lang='en',
                    use_space_char=True,
                    show_log=False,
                    enable_mkldnn=True)

    return ocr

def minify_boxes(boxes):
    """Turn a 4-points box into a 2-points box"""
    ret = []
    for box in boxes:
        (a1, a2), (b1, b2), (c1, c2), (d1, d2) = box
        ret.append([int(min(a1, b1, c1, d1)),
                    int(min(a2, b2, c2, d2)),
                    int(max(a1, b1, c1, d1)),
                    int(max(a2, b2, c2, d2))])
    return ret

def run_ocr(image):
    result = ocr.ocr(image)[0] if isinstance(image, str) else ocr.ocr(np.array(image))[0]
    if result is None:
        return {'words': [], 'boxes': [], 'text': ''}
    return {
        'words': [line[1][0] for line in result],
        'boxes': [line[0] for line in result],
        'text': ' '.join([line[1][0] for line in result])
    }

def process_document(image):
    # Step 1: OCR
    ocr_output = run_ocr(image)
    img = Image.open(image).convert('RGB') if isinstance(image, str) else image
    # Step 2: LayoutLMv3 encoding
    inputs = layoutlm_processor(img, ocr_output['words'],
                                boxes=minify_boxes(ocr_output['boxes']),
                                return_tensors="pt")
    outputs = layoutlm_model(**inputs)
    layoutlm_embeddings = outputs.last_hidden_state

    return layoutlm_embeddings, ocr_output['text']

ocr = load_PaddleOCR()
layoutlm_processor, layoutlm_model = load_LayoutLMv3()