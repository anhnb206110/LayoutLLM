import os
import torch
import logging
import argparse
import torch.nn as nn
from data.data_loader import get_train_dataset, get_val_dataset
from models.utils import print_number_of_trainable_model_parameters
from transformers import Trainer, TrainingArguments

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, device="cuda"):
        question_ids = inputs['question_ids'].squeeze(0)
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        bbox = inputs['bbox'].squeeze(0)
        pixel_values = inputs['pixel_values'].squeeze(0)
        labels = inputs['labels'].squeeze(0)
        outputs = model(question_ids=question_ids,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        bbox=bbox,
                        pixel_values=pixel_values)
        
        shifted_labels = labels[:, 1:].contiguous()
        shifted_logits = outputs[:, :-1, :].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=llm_pipe.tokenizer.eos_token_id)

        loss = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def frozen(layout_llm_model):
    frozen_parts = ['dtm', 'embed', 'llm']
    for part in frozen_parts:
        for param in layout_llm_model.__getattr__(part).parameters():
            param.requires_grad = False
    active_parts = ['visual_projector', 'text_projector']
    for part in active_parts:
        for param in layout_llm_model.__getattr__(part).parameters():
            param.requires_grad = True
    return layout_llm_model

def main(args):
    global layout_llm, llm_pipe, llm_model
    from models.LayoutLLM import get_model
    layout_llm, llm_pipe, _ = get_model(args.llm, device=args.device, token=args.hf_token)
    llm_model = llm_pipe.model

    layout_llm = frozen(layout_llm)

    num_epochs = args.num_epochs
    batch_size = 1
    train_params = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        # evaluation_strategy="steps",
        # eval_steps=100,
        gradient_accumulation_steps=1,
        optim=args.optim,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        remove_unused_columns=False
    )

    fine_tuning = CustomTrainer(
        model=layout_llm,
        args=train_params,
        train_dataset=get_train_dataset(args.data_path),
        eval_dataset=get_val_dataset(args.data_path),
    )

    print(print_number_of_trainable_model_parameters(layout_llm))

    fine_tuning.train()
    torch.save(fine_tuning.model.state_dict(), os.sep.join([args.output_dir, "layout_llm.pt"]))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./processed_data", help="Data path")
    parser.add_argument('--llm', type=str, default="meta-llama/Llama-2-7b-chat-hf", help="LLM-pretrained model")
    parser.add_argument('--dtm', type=str, default="microsoft/layoutlmv3-base", help="Doc-pretrained model")
    parser.add_argument('--output_dir', type=str, default='./output', help="Output dir for checkpoints and last model")
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--optim', type=str, default="adamw_torch")
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--logging_steps', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
    parser.add_argument('--hf_token', type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    print(args)
    main(args)