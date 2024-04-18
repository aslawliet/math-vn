from core.dataseter import DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN
from core.dataseter import OVMDataset, OVMDataCollator, get_dataloader

from core.utils import (
    get_optimizer, get_scheduler, disable_model_dropout, clip_model_gradients, get_all_reduce_mean,
    should_run_eval, evaluation, save_model
)

from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, MixedPrecision)
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import AutoModelForSequenceClassification
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
import functools, wandb, uuid, torch, transformers, os, huggingface_hub
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm

wandb_key = "YOUR_WANDB_API_KEY"
huggingface_token = "YOUR_HF_TOKEN"

wandb.login(key=wandb_key)
huggingface_hub.login(token=huggingface_token)

def setup_model(model_name, max_length):
    model =  AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=model_name,
        num_labels=1,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        pad_token=DEFAULT_PAD_TOKEN,
        trust_remote_code=True
    )


    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer




def log_stats(pbar, wandb, epoch, loss_tensor, grad_norm, scheduler):
    last_lr = scheduler.get_last_lr()[0]

    wandb.log(
        {
            "current_loss": loss_tensor,
            "current_epoch": epoch,
            "learning_rate": last_lr,
            "grad_norm": grad_norm,
        },
    )

    current_loss = f"{loss_tensor:.4f}"
    current_lr = f"{last_lr:.10f}"

    pbar.set_description(f"Epoch {epoch:.2f}, Loss: {current_loss}, LR: {current_lr}")


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)

    # model_name = "mistralai/Mistral-7B-v0.1"
    model_name = "ruliad/ovm-base-m-3"

    scheduler_type = "cosine"
    seed = 877645  # set your seed
    transformers.set_seed(seed)

    run_id = str(uuid.uuid4())
    output_dir = f"./outputs/{model_name}/{run_id}"
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I_%M_%S_%p")
    max_length = 4196  # adjust as needed
    disable_dropout = False
    gradient_checkpointing = True
    clip_gradients = True
    shuffle = True  # multipack sampler already does random sampling
    train_batch_size = 8  # adjust as needed
    validation_batch_size = 1  # adjust as needed
    epochs = 3  # adjust as needed
    lr = 2e-06  # adjust as needed
    weight_decay = 0.0  # adjust as needed
    gradient_clipping = 1.0  # adjust as needed
    train_on_inputs = False  # whether to train on instruction tokens
    use_multipack_sampler = (
        False  # whether to use the multipack sampler or torch sampler
    )

    model, tokenizer = setup_model(model_name, max_length)
    num_params = sum([p.numel() for p in model.parameters()])
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            MistralDecoderLayer,
        },
    )
    from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, ShardingStrategy

    fsdp_config = dict(
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        # use_orig_params=False,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=None,
        param_init_fn=None,
        cpu_offload=CPUOffload(offload_params=True),
        sync_module_states=True,
    )

    model = FSDP(model, **fsdp_config)
    optimizer = get_optimizer(model, lr, weight_decay)


    train_ds = ["data/train.jsonl"]
    val_ds = ["data/validation.jsonl"]
    train_dataset = OVMDataset(train_on_inputs, tokenizer, train_ds)
    val_dataset = OVMDataset(train_on_inputs, tokenizer, val_ds)
    collator = OVMDataCollator(tokenizer)
    pos_weights = torch.tensor(train_dataset.pos_weights)
    neg_weights = torch.tensor(train_dataset.neg_weights)
    print(f"POSITIVE WEIGHTS: {pos_weights}")
    print(f"NEGATIVE WEIGHTS: {neg_weights}")
    train_sampler, train_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        train_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        train_batch_size,
    )
    val_sampler, val_loader = get_dataloader(
        use_multipack_sampler,
        max_length,
        val_dataset,
        world_size,
        local_rank,
        shuffle,
        seed,
        collator,
        validation_batch_size,
    )
 

    if use_multipack_sampler:
        total_steps_per_epoch = train_sampler.num_batches()
    else:
        total_steps_per_epoch = len(train_loader)

    max_steps = total_steps_per_epoch * epochs
    scheduler = get_scheduler(local_rank, scheduler_type, optimizer, max_steps)

    if local_rank == 0:
        run = wandb.init(
            project="mistral-7b",
            name=run_id,
            config={
                "model_name": model_name,
                "run_id": run_id,
                "date": date_of_run,
                "dataset_size": len(train_dataset),
                "dataset": ",".join(train_ds),
                "validation": ",".join(val_ds),
                "weight_decay": weight_decay,
                "clip_gradients": clip_gradients,
                "learning_rate": lr,
                "shuffle": shuffle,
                "seed": seed,
                "disable_dropout": disable_dropout,
                "use_multipack_sampler": use_multipack_sampler,
                "train_on_inputs": train_on_inputs,
                "epochs": epochs,
                "acc_steps": acc_steps,
                "batch_size": train_batch_size,
                "total_batch_size": train_batch_size * world_size,
                "scheduler_type": scheduler_type,
            },
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if disable_dropout:
        disable_model_dropout(model)

    model.train()
    dist.barrier()

    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        current_epoch = epoch + 1

        pbar = tqdm(
            enumerate(train_loader),
            total=total_steps_per_epoch,
            colour="green",
            desc=f"Epoch {epoch}.00",
            disable=(local_rank != 0),
        )

        for step, batch in pbar:
            current_step = step + 1

            inputs = {
                "input_ids": batch["input_ids"].to(model.device),
                "labels": batch["labels"].to(model.device),
                "attention_mask": batch["attention_mask"].to(model.device)
            }
            
            

            # forward
            outputs = model(**inputs)
            loss = outputs.loss

            # backward
            loss.backward()

            # clipping
            if clip_gradients:
                grad_norm = clip_model_gradients(model, gradient_clipping)

            # weight update
            optimizer.step()
            scheduler.step()

            # zero gradients after weight update
            optimizer.zero_grad(set_to_none=True)

            # detach from graph
            loss = loss.detach()

            # avg loss over all processes
            loss = get_all_reduce_mean(loss).item()

            if local_rank == 0:
                log_stats(
                    pbar,
                    wandb,
                    round((current_step / total_steps_per_epoch), 2) + epoch,
                    loss,
                    grad_norm,
                    scheduler,
                )

            # runs eval 2x an epoch, adjust as needed
            if should_run_eval(total_steps_per_epoch, 2, current_step):
                validation_loss = evaluation(
                    model,
                    val_loader,
                    wandb,
                    local_rank,
                )

                save_model(
                    local_rank,
                    model,
                    tokenizer,
                    output_dir,
                    current_epoch,
                    current_step,
                )

                model.train()

    # save final model
    save_model(local_rank, model, tokenizer, output_dir, epochs, "final")
