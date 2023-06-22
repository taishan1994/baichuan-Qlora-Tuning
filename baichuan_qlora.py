import os
import torch
import bitsandbytes as bnb
import numpy as np

from dataset import load_data, NerCollate

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer

model_cards = {
    "baichuan-7b": (AutoTokenizer, AutoModelForCausalLM),
}


def get_data():
    train_dataset = load_data(args.train_path)
    eval_dataset = load_data(args.eval_path)
    return train_dataset, eval_dataset


def get_model(args):
    device_map = {"": 0}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 trust_remote_code=True,
                                                 quantization_config=BitsAndBytesConfig(
                                                     load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                                     bnb_4bit_use_double_quant=True,
                                                     bnb_4bit_quant_type='nf4'
                                                 ),
                                                 device_map=device_map)
    model = prepare_model_for_kbit_training(model)
    return tokenizer, model


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    # 4BIT进行训练，这里要除以2
    trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

class Args:
    model_name = "baichuan"
    model_path = "model_hub/baichuan-7B"
    output_dir = "./checkpoint/{}/".format(model_name)
    max_seq_length = 128 + 64
    instruct_column = "instruct"
    query_column = "query"
    response_column = "answer"
    train_path = "data/msra/train.txt"
    eval_path = "data/msra/eval.txt"
    resume_from_checkpoint = False
    bf16 = False
    eval = False

args = Args()

tokenizer, model = get_model(args)
"""
modules = find_all_linear_names(model)
for module in modules:
    print(module)
    
W_pack
up_proj
o_proj
down_proj
gate_proj
"""

modules = ["W_pack"]

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=modules,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
for name, module in model.named_modules():
    if isinstance(module, LoraLayer):
        if args.bf16:
            module = module.to(torch.bfloat16)
    if 'norm' in name:
        module = module.to(torch.float32)
    if 'lm_head' in name or 'embed_tokens' in name:
        if hasattr(module, 'weight'):
            if args.bf16 and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)
        
model.config.use_cache = False
print_trainable_parameters(model)

resume_from_checkpoint = args.resume_from_checkpoint

if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")


train_dataset, eval_dataset = get_data()
collate = NerCollate(args, tokenizer)

train_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=64,
    learning_rate=3e-4,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps" if args.eval else "no",
    save_strategy="steps",
    eval_steps=70 if args.eval else None,
    save_steps=70,
    logging_steps=10,
    output_dir=args.output_dir,
    report_to="tensorboard",
    save_total_limit=1,
    overwrite_output_dir=True,
    load_best_model_at_end=True if args.eval else False,
)


class PeftTrainer(Trainer):
    def _save_checkpoint(self, _, trial, metrics=None):
        """ Don't save base model, optimizer etc.
            but create checkpoint folder (needed for saving adapter) """
        pass
            
class PeftSavingCallback(TrainerCallback):
    """ Correctly save PEFT model and not full model """

    def _save(self, model, folder):
        if folder is None:
            folder = args.output_dir
        peft_model_path = os.path.join(folder, "adapter_model")
        model.save_pretrained(peft_model_path)

    def on_save(self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
                ):

        checkpoint_folder = args.output_dir

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """ Save final best model adapter """
        self._save(kwargs['model'], state.best_model_checkpoint)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """ Save intermediate model adapters in case of interrupted training """
        # folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self._save(kwargs['model'], args.output_dir)     
        
        
trainer = PeftTrainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate.collate_fn,
    callbacks=[PeftSavingCallback],
)

trainer.train()

model.save_pretrained(os.path.join(args.output_dir, "adapter_model"))
