from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("vit5-large-vietnews-summarization")  
model = AutoModelForSeq2SeqLM.from_pretrained("vit5-large-vietnews-summarization")
model.to(device)



from newspaper import Config
from newspaper import Article

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
url = "https://thanhnien.vn/quan-an-sang-khu-cho-lon-voi-gia-chi-15000-20000-dong-khach-den-dong-nghet-185230917165947819.htm"

config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10
#get all paper
# for article in paper.articles:
#     print(article.url)

article = Article(url, config=config)
article.download()
article.parse()
# the replace is used to remove newlines
article_text = article.text.replace('\n', ' ')
#print(article_text)

# sentence = '''

# '''
text =  article_text
encoding = tokenizer(text, return_tensors="pt").to(device)
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
outputs = model.generate(
    input_ids=input_ids, attention_mask=attention_masks,
    max_length=1000,
    early_stopping=True
)
for output in outputs:
    line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(line)