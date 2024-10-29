from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
    AddedToken,
)
from tokenizers.normalizers import Normalizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset


# initialize tokenizer
tokenizer = Tokenizer(models.BPE(unk_token=None, fuse_unk=False, dropout=None,
                      end_of_word_suffix="", continuing_subword_prefix="", byte_fallback=True,))


# normalizers
normalizer_sequence = normalizers.Sequence([
    normalizers.NFC(),
])


# pretokenizer
pretokenizer_sequence = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior="isolated", invert=False),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
])


# decoder
decoder = decoders.ByteLevel()


# special tokens
cls = AddedToken(content="[CLS]", lstrip=False,
                 normalized=False, rstrip=False, single_word=True)
pad = AddedToken(content="[PAD]", lstrip=False,
                 normalized=False, rstrip=False, single_word=True)
mask = AddedToken(content="[MASK]", lstrip=False,
                  normalized=False, rstrip=False, single_word=True)
sep = AddedToken(content="[SEP]", lstrip=False,
                 normalized=False, rstrip=False, single_word=True)
unk = AddedToken(content="[UNK]", lstrip=False,
                 normalized=False, rstrip=False, single_word=True)

# create dataset and initialize generator
dataset = load_dataset(
    "../data_all/data", streaming=True, spilt="train")
batch_size = 15000


def get_training_corpus():
    content = []
    for row in dataset:
        if len(content) == batch_size:
            yield content
            content = []
        else:
            content.append(row["text"])

    if content:
        yield content


training_corpus = get_training_corpus()


# add modules
tokenizer.normalizer = normalizer_sequence
tokenizer.pre_tokenizer = pretokenizer_sequence
tokenizer.decoder = decoder


# initialize vocab size and trainer
# (30000 + 261 + 64 - 53)
vocab_size = 30272
max_token_length = 2048
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size, show_progress=True, max_token_length=max_token_length)


# start training process
tokenizer.train_from_iterator(training_corpus, trainer=trainer)


# add special tokens and more configurations
tokenizer.add_special_tokens([cls, pad, mask, sep, unk])
tokenizer.enable_padding(
    pad_id=vocab_size+1, pad_type_id=vocab_size+1, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=max_token_length)


# add preprocessors
postprocessor_sequence = processors.Sequence([
    processors.ByteLevel(trim_offsets=False),
    processors.TemplateProcessing(
        single="[CLS] $0 [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab_size), ("[SEP]", vocab_size + 3)]),
])

tokenizer.post_processor = postprocessor_sequence


# convert to fast tokenizer
tokenizer.save("./trained_tokenizer.json")
wrapped_tokenizer = PreTrainedTokenizerFast(
    # tokenizer,
    tokenizer_file="./trained_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
wrapped_tokenizer.model_max_length = max_token_length
wrapped_tokenizer.save_pretrained("./tokenizer", legacy_format=False)
