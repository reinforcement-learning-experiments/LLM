
# --- LLM

!pip install -q -U transformers peft accelerate optimum bitsandbytes
!pip install -q -U huggingface_hub
!huggingface-cli login

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TextStreamer

name = 'microsoft/phi-2'

model=AutoModelForCausalLM.from_pretrained(name, low_cpu_mem_usage=True, load_in_8bit=True)
tokenizer=AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id=tokenizer.eos_token_id

generation_config=GenerationConfig(max_new_tokens=500,
                                    temperature=0.4,
                                    top_p=0.95,
                                    top_k=40,
                                    repetition_penalty=1.2,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    do_sample=True,
                                    use_cache=True,
                                    output_attentions=False,
                                    output_hidden_states=False,
                                    output_scores=False,
                                    remove_invalid_values=True
                                    )
streamer = TextStreamer(tokenizer)

def ask_model(instruction, system='Du bist ein hilfreicher Assistent.'):
    prompt=f"{system} USER: {instruction} ASSISTANT:"
    input_tokens=tokenizer(prompt, return_tensors="pt").to(model.device)
    output_tokens=model.generate(**input_tokens,  generation_config=generation_config, streamer=streamer)[0]
    answer=tokenizer.decode(output_tokens, skip_special_tokens=True)
    return answer

answer=ask_model('Wenn du 2024 als Bundeskanzler kandidieren würdest, was wäre dein Wahlprogramm?')

# Example for Factual/Retrieval with less Hallucinations

Please use the Format and system message below for Factual retrieval. The model is trained to only answer from provided context and not hallucinate any answers or additional information.

retrieval_system="Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese wortwörtlich mit einem Zitat aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist!"
retrieval_question="""\
BEGININPUT
BEGINCONTEXT
Url: https://www.jph.me
ENDCONTEXT
Das Wetter in Düsseldorf wird heute schön und sonnig!
ENDINPUT
BEGININSTRUCTION Was ist 1+1? ENDINSTRUCTION"""
ask_model(retrieval_question, system=retrieval_system)

retrieval_system="Du bist ein hilfreicher Assistent. Für die folgende Aufgabe stehen dir zwischen den tags BEGININPUT und ENDINPUT mehrere Quellen zur Verfügung. Metadaten zu den einzelnen Quellen wie Autor, URL o.ä. sind zwischen BEGINCONTEXT und ENDCONTEXT zu finden, danach folgt der Text der Quelle. Die eigentliche Aufgabe oder Frage ist zwischen BEGININSTRUCTION und ENDINCSTRUCTION zu finden. Beantworte diese wortwörtlich mit einem Zitat aus den Quellen. Sollten diese keine Antwort enthalten, antworte, dass auf Basis der gegebenen Informationen keine Antwort möglich ist!"
retrieval_question="""\
BEGININPUT
BEGINCONTEXT
Url: https://www.jph.me
ENDCONTEXT
Das Wetter in Düsseldorf wird heute schön und sonnig!
ENDINPUT
BEGININSTRUCTION Wie wird das Wetter heute in Düsseldorf? ENDINSTRUCTION"""
ask_model(retrieval_question, system=retrieval_system)

# --- Q/A:

## Tutorial: Build Your First Question Answering System
https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline
### Q&A Model:
https://huggingface.co/deepset/roberta-base-squad2

from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
# model_name = "timpal0l/mdeberta-v3-base-squad2"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

QA_input = {
    'question': 'Was ist Wikipedia?',
    'context': 'Wikipedia ist ein Projekt zum Aufbau einer Enzyklopädie aus freien Inhalten, zu denen du sehr gern beitragen kannst. Seit März 2001 sind 2.874.691 Artikel in deutscher Sprache entstanden.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

res



