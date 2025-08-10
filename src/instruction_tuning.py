import time

import tiktoken
import torch

from src.config.config import BASE_CONFIG, MODEL_CONFIGS
from src.dl_it_data import download_and_load_file
from src.generate_text import text_to_token_ids, token_ids_to_text
from src.gpt_download import download_and_load_gpt2
from src.models.dummy_gpt_model import GPTModel
from src.models.loss import calc_loss_loader
from src.postprocess.postprocess import generate
from src.preprocess.assign_pretrained_gpt2 import load_weights_into_gpt
from src.preprocess.format_prompt_for_it import (
    create_instruction_tuning_dataset,
    format_input,
)
from src.train import plot_losses, train_model_simple

if __name__ == "__main__":
    file_path = "./output/instruction_data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)
    print("number of entries:", len(data))
    idx = 999
    print(data[idx])

    formatted_prompt = format_input(data[idx])
    print(formatted_prompt + f"\n\n### Response:\n{data[idx]['output']}")

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    print("Training data:", len(train_data))
    print("Validation data:", len(val_data))
    print("Testing data:", len(test_data))

    tokenizer = tiktoken.get_encoding("gpt2")
    train_data_loader, val_data_loader, test_data_loader = create_instruction_tuning_dataset(
        data,
        tokenizer=tokenizer,
        train_ratio=0.85,
        test_ratio=0.1,
        num_workers=0,
        batch_size=8,
        allowed_max_length=1024,
        device="cpu",
    )

    settings, params = download_and_load_gpt2("355M", "./output/gpt2")
    print("settings:", settings)
    print("params:", params.keys())

    BASE_CONFIG.update(MODEL_CONFIGS["gpt2-medium(355M)"])
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    train_data_loader, val_data_loader, test_data_loader = create_instruction_tuning_dataset(
        data,
        tokenizer=tokenizer,
        train_ratio=0.85,
        test_ratio=0.1,
        num_workers=0,
        batch_size=8,
        allowed_max_length=1024,
        device="cpu",
    )

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print("Input text:", input_text)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=int(BASE_CONFIG["context_length"]),
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text) :].strip()
    print("Generated response:", response_text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_data_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_data_loader, model, device, num_batches=5)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_data_loader,
        val_data_loader,
        optimizer,
        num_epochs=num_epochs,
        device=device,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "./output/gpt2_medium_model_and_adamw.pth",
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
