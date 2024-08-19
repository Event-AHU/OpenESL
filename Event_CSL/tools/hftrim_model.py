from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
import pickle
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer
import gzip
import pickle


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

file = 'pkl path'
raw_data = load_dataset_file(file)

text = []
char_text = []
char_vocab = []

for key,value in raw_data.items():
    sentence = value['text']
    text.append(sentence)
    char_text.append(' '.join(list(sentence)))
    char_vocab.extend(list(sentence))

char_vocab = list(set(char_vocab))

tokenizer = MBartTokenizer.from_pretrained("mbart-large-cc25", src_lang="zh_CN", tgt_lang="zh_CN")
bos_index = tokenizer.convert_tokens_to_ids('<s>')
pad_index = tokenizer.convert_tokens_to_ids('<pad>')
eos_index = tokenizer.convert_tokens_to_ids('</s>')
unk_index = tokenizer.convert_tokens_to_ids('<unk>')

model = MBartForConditionalGeneration.from_pretrained("mbart-large-cc25")
configuration = model.config

# trim tokenizer
tt = TokenizerTrimmer(tokenizer)

tt.make_vocab(char_text)
tt.make_tokenizer()

# trim model
mt = MBartTrimmer(model, configuration, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()

new_tokenizer = tt.trimmed_tokenizer
new_model = mt.trimmed_model

new_tokenizer.save_pretrained('/pretrain_lm/hftrim_mbart')
new_model.save_pretrained('/pretrain_lm/hftrim_mbart')

# sl_mbart
configuration = MBartConfig.from_pretrained('/pretrain_lm/config.json')
configuration.vocab_size = new_model.config.vocab_size
mytran_model = MBartForConditionalGeneration._from_config(config=configuration)
mytran_model.model.shared = new_model.model.shared

mytran_model.save_pretrained('/pretrain_lm/sl_mbart')