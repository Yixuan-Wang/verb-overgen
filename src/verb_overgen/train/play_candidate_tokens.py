import einops as eo
import IPython
from jaxtyping import Float
import rich
from rich.prompt import Prompt
from rich.table import Table
import torch
import transformers as hf
import tokenizers as hftk

from verb_overgen.train.train import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, MAX_LEN, VerbOvergenTrainModule

torch.set_default_device("cuda")

# state = torch.load("/u501/y4466wan/projects/2025/verb-overgen/data/train/gpt2/verb-overgen-gpt2-a6cbd4_vital-microwave-3_final.pth", weights_only=True, map_location="cpu")
tokenizer = hf.PreTrainedTokenizerFast(
    tokenizer_object=hftk.Tokenizer.from_file("/u501/y4466wan/projects/2025/verb-overgen/data/tokenizer/1024-wordlevel/tokenizer.json"),
)

module = VerbOvergenTrainModule.load_from_checkpoint("/u501/y4466wan/projects/2025/verb-overgen/data/train/gpt2/verb-overgen-gpt2-a6cbd4_silver-salad-4_final.pth", map_location="cuda")
model = module.model

model.eval()
torch.no_grad().__enter__()

bos_token_id = tokenizer._tokenizer.token_to_id(BOS_TOKEN)

pronouns = ["i", "you", "he", "she", "they", "we"]

input_ids = torch.tensor([[bos_token_id, *tokenizer._tokenizer.encode(input).ids] for input in pronouns])
output = model(input_ids=input_ids)
logits: Float[torch.Tensor, "batch seq vocab"] = output.logits  #
logits_last: Float[torch.Tensor, "batch vocab"] = logits[:, -1, :]

token_probs = torch.softmax(logits_last, dim=-1).sort(dim=-1, descending=True)

table = Table(title="Top Candidate Tokens by Pronoun")
table.add_column("Pronoun", justify="left", style="cyan", no_wrap=True)
for i in range(10):
    table.add_column(f"Rank {i+1}", justify="left", style="magenta")

for i, pronoun in enumerate(pronouns):
    top_tokens = [
        "{token}({prob:.3%})".format(
            token=tokenizer._tokenizer.id_to_token(int(token_id)),
            prob=token_prob.item()
        ) for token_id, token_prob in
        zip(token_probs.indices[i, :10], token_probs.values[i, :10])
    ]
    table.add_row(pronoun, *top_tokens)

top_tokens = set[str]()
for i, pronoun in enumerate(pronouns):
    top_tokens.update({
        tokenizer._tokenizer.id_to_token(int(token_id))
        for token_id in token_probs.indices[i, :50]
    })