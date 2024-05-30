import os
import torch
import argparse
from models.utils import pad_input_ids, pad_bbox
from models.ocr import run_ocr, minify_boxes, layoutlm_processor
from models.llm import load_LLaMa_model
from models.LayoutLLM import get_model
from PIL import Image

def layout_llm_generate(model, image, question, max_length=20, **kwargs):
    device = kwargs.get('device', 'cuda')
    model.to(device)
    model.eval()
    image = Image.open(image).convert('RGB')
    image = image.convert('RGB')
    aspect = image.width / image.height
    if aspect >= 1:
        image = image.resize((1000, int(1000 / aspect)))
    else:
        image = image.resize((int(1000 * aspect), 1000))
    ocr_output = run_ocr(image)
    boxes = minify_boxes(ocr_output['boxes'])
    processed = layoutlm_processor(image, ocr_output['words'],
                                    boxes=boxes,
                                    return_tensors="pt")
    question_ids = llm_pipe.tokenizer.encode(question,
                                             return_tensors="pt",
                                             padding="max_length",
                                             max_length=512).to(device)
    attention_mask = pad_input_ids(processed.attention_mask, 315, 0).to(device)
    input_ids = pad_input_ids(processed.input_ids, 315, 2).to(device)
    bbox = pad_bbox(processed.bbox, 315).to(device)
    pixel_values = processed.pixel_values.to(device)
    for _ in range(max_length):
        outputs = model(question_ids=question_ids,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        pixel_values=pixel_values)
        next_token_logits = outputs[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        if next_token_id == llm_pipe.tokenizer.eos_token_id:
            break
        question_ids = torch.cat([question_ids, next_token_id.unsqueeze(-1)], dim=1).to(device)

    output_text = llm_pipe.tokenizer.decode(question_ids[0], skip_special_tokens=True)
    return output_text, outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--hf_token', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()
    llm_pipe = load_LLaMa_model(args.llm, token=args.hf_token)
    layout_llm, llm_pipe, layoutlm_model = get_model(args.llm, token=args.hf_token)
    layout_llm.load_state_dict(torch.load(os.sep.join(['./output', 'layoutlm.pt'])))
    layout_llm_generate(layout_llm, args.image, args.question)