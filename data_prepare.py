from models.utils import pad_input_ids, pad_bbox
from models.ocr import run_ocr, minify_boxes, layoutlm_processor
from models.llm import load_LLaMa_model
from PIL import Image
from datasets import load_dataset, DatasetDict

import argparse

def get_text_box(x):
    image = Image.open(x['image'][0]).convert('RGB')
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
    x['input_ids'] = pad_input_ids(processed.input_ids, 315, llm_pipe.tokenizer.pad_token_id)  # ids of text in image
    x['question_ids'] = llm_pipe.tokenizer.encode("[INST] " + x['messages'][0]['content'] + "[/INST]",
                                                  return_tensors="pt",
                                                  padding="max_length",
                                                  max_length=512)
    x['attention_mask'] = pad_input_ids(processed.attention_mask, 315, 0)
    x['bbox'] = pad_bbox(processed.bbox, 315)  # bounding boxes
    x['pixel_values'] = processed.pixel_values  # the image after preprocess
    x['image'] = image
    x['labels'] = llm_pipe.tokenizer.encode(x['messages'][-1]['content'] + '</s>',
                                            return_tensors="pt",
                                            add_special_tokens=False,
                                            padding="max_length",
                                            max_length=1024)
    return x

def prepare_data(args):
    llm_pipe.tokenizer.pad_token_id = llm_pipe.tokenizer.eos_token_id
    training_data = load_dataset('mPLUG/DocLocal4K', split="train")
    if args.sample:
        training_data = training_data.shuffle().select(range(20))
    tokenized_data = training_data.map(get_text_box)
    tokenized_data = tokenized_data.remove_columns(['image', 'messages', 'task_name', 'dataset_name', 'id'])
    train_test_split = tokenized_data.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'val': val_dataset
    })

    dataset_dict.save_to_disk('processed_data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--hf_token', type=str, required=True)
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args()
    llm_pipe = load_LLaMa_model(args.llm, access_token=args.hf_token)
    prepare_data(args)
    