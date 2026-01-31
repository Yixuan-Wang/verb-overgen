import einops as eo
import IPython
from jaxtyping import Float
import rich
from rich.prompt import Prompt
from rich.table import Table
import torch
import transformers as hf
import tokenizers as hftk

from verb_overgen.train.train import VerbOvergenTrainModule
from verb_overgen.common.tokenize import BOS_TOKEN, EOS_TOKEN

torch.set_default_device("cuda")

# state = torch.load("/u501/y4466wan/projects/2025/verb-overgen/data/train/gpt2/verb-overgen-gpt2-a6cbd4_vital-microwave-3_final.pth", weights_only=True, map_location="cpu")
tokenizer = hf.PreTrainedTokenizerFast(
    tokenizer_object=hftk.Tokenizer.from_file("./data/tokenizer/1024-wordlevel/tokenizer.json"),
)

module = VerbOvergenTrainModule.load_from_checkpoint("./var/train/20260128/silver-durian-35/checkpoints/silver-durian-35-epoch=02.ckpt", map_location="cuda")
model = module.model

# model = hf.GPT2LMHeadModel(config=hf.GPT2Config(
#     vocab_size=tokenizer.vocab_size,
#     n_positions=MAX_LEN,
#     n_ctx=MAX_LEN,
#     n_embd=768,
#     n_layer=12,
#     n_head=12,
#     pad_token_id=tokenizer._tokenizer.token_to_id(PAD_TOKEN),
#     bos_token_id=tokenizer._tokenizer.token_to_id(BOS_TOKEN),
#     eos_token_id=tokenizer._tokenizer.token_to_id(EOS_TOKEN),
# ))

# class Dummy(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

# dummy = Dummy(model)
# dummy.load_state_dict(state["state_dict"])
# model = dummy.model

model.eval()
torch.no_grad().__enter__()


def loop():
    bos_token_id = tokenizer._tokenizer.token_to_id(BOS_TOKEN)
    while next_step := Prompt.ask("Enter prompt (or 'exit' to quit): "):
        if next_step == "exit":
            break

        IPython.get_ipython().run_line_magic('clear', '') # type: ignore
        print(f"Prompt: {next_step}")
        input = tokenizer._tokenizer.encode(next_step)
        input_ids = torch.tensor([[bos_token_id, *input.ids]])
        input_tokens = input.tokens
        
        rich.print(f"Input tokens: {input_tokens}")

        output = model(input_ids=input_ids)
        logits: Float[torch.Tensor, "1 seq vocab"] = output.logits
        token_prob = torch.softmax(logits[0, -1], dim=-1).sort(descending=True)

        table = Table("rank", "prob", "token_id", "token")
        for rank, (prob, token_id) in enumerate(zip(token_prob.values[:10], token_prob.indices[:10]), start=1):
            token = tokenizer._tokenizer.id_to_token(int(token_id))
            table.add_row(str(rank), f"{prob.item():.3%}", str(token_id.item()), token)

        rich.print(table)

def compare():
    bos_token_id = tokenizer._tokenizer.token_to_id(BOS_TOKEN)
    eos_token_id = tokenizer._tokenizer.token_to_id(EOS_TOKEN)

    while True:
        prompt_1 = Prompt.ask("The first prompt")
        if not prompt_1 or prompt_1 == "exit":
            break

        prompt_2 = Prompt.ask("The second prompt")
        if not prompt_2 or prompt_2 == "exit":
            break

        IPython.get_ipython().run_line_magic('clear', '') # type: ignore
        print(f"Prompt 1: {prompt_1}")
        print(f"Prompt 2: {prompt_2}")
        input_1 = tokenizer._tokenizer.encode(prompt_1)
        input_2 = tokenizer._tokenizer.encode(prompt_2)
        input_ids_1: Float[torch.Tensor, "1 seq_1"] = torch.tensor([[bos_token_id, *input_1.ids, eos_token_id]])
        input_ids_2: Float[torch.Tensor, "1 seq_2"] = torch.tensor([[bos_token_id, *input_2.ids, eos_token_id]])
        print(f"Input tokens 1: {input_1.tokens}")
        print(f"Input tokens 2: {input_2.tokens}")

        output_1 = model(input_ids=input_ids_1)
        output_2 = model(input_ids=input_ids_2)
        logits_1: Float[torch.Tensor, "1 seq_1 vocab"] = output_1.logits
        logits_2: Float[torch.Tensor, "1 seq_2 vocab"] = output_2.logits

        prob_1: Float[torch.Tensor, "1 seq_1 vocab"] = torch.softmax(logits_1, dim=-1)
        prob_2: Float[torch.Tensor, "1 seq_2 vocab"] = torch.softmax(logits_2, dim=-1)


        token_prob_1 = torch.gather(
            prob_1[:, :-1, :],
            dim=-1,
            index=eo.rearrange(input_ids_1[:, 1:], "1 seq -> 1 seq 1"),
        )
        token_prob_2 = torch.gather(
            prob_2[:, :-1, :],
            dim=-1,
            index=eo.rearrange(input_ids_2[:, 1:], "1 seq -> 1 seq 1"),
        )

        seq_prob_1 = token_prob_1.prod()
        seq_prob_2 = token_prob_2.prod()

        print(f"Sequence probability 1: {seq_prob_1.item():.6e}")
        print(f"Sequence probability 2: {seq_prob_2.item():.6e}")

        print(f"Token-level probabilities 1: {[f'{p.item():.6e}' for p in token_prob_1[0, :, 0]]}")
        print(f"Token-level probabilities 2: {[f'{p.item():.6e}' for p in token_prob_2[0, :, 0]]}")

def prefix_inspection():
    bos_token_id = tokenizer._tokenizer.token_to_id(BOS_TOKEN)
    eos_token_id = tokenizer._tokenizer.token_to_id(EOS_TOKEN)

    jane: int = tokenizer._tokenizer.token_to_id("jane")

    first_token_probs = torch.zeros((32 + 1, 32 + 1))
    for length in range(1, 32 + 1):
        input = [bos_token_id, *(jane for _ in range(length)), eos_token_id]
        input_ids: Float[torch.Tensor, "1 seq"] = torch.tensor([input])
        output = model(input_ids=input_ids)
        logits: Float[torch.Tensor, "1 seq vocab"] = output.logits
        prob: Float[torch.Tensor, "1 seq vocab"] = torch.softmax(logits, dim=-1)
        token_prob = torch.gather(
            prob[:, :-1, :],
            dim=-1,
            index=eo.rearrange(input_ids[:, 1:], "1 seq -> 1 seq 1"),
        )
        token_prob: Float[torch.Tensor, "seq"] = token_prob[0, :, 0]
        first_token_probs[length, :len(token_prob)] = token_prob
        # first_token_probs.append(first_token_prob)

    return first_token_probs

if __name__ == "__main__":
    loop()
    # compare()
    # prefix_inspection()    