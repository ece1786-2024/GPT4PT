from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import torch
import os

class TfClassifier:
    '''
    Prerequisite:
    1. A `tokenizer` folder under the same directory to be loaded by `AutoTokenizer`
    2. A `model` folder under the same directory to be loaded by `AutoModelForSequenceClassification`
    3. `OPENAI_API_KEY` set in `os.environ`
    '''

    def __init__(self):
        self.labels = {0: "T", 1: "F"}
        self.tokenizer = None
        self.gpt2 = None
        self.gpt4 = OpenAI()
        
        self._load_tokenizer()
        self._load_gpt2()

    def _load_tokenizer(self):
        tokenizer_path = "tokenizer"
        assert(os.path.exists(tokenizer_path))
        #print("Loading existing tokenizer from:", tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _load_gpt2(self):
        model_path = 'TF'
        assert(os.path.exists(model_path))
        #print("Loading existing model from:", model_path)
        self.gpt2 = AutoModelForSequenceClassification.from_pretrained(model_path)

    def _init_gpt4(self):
        prompt_path = 'TF/prompt.txt'
        #assert(['OPENAI_API_KEY'] in os.environ)
        assert 'OPENAI_API_KEY' in os.environ, "OPENAI_API_KEY is not set in the environment variables"
        assert(os.path.exists(prompt_path))

        
        with open(prompt_path, 'r') as file:
            prompt = file.read()
        # some api version doens't support memory, so no need to init.
        #    #print(prompt)
        #    _ = self.gpt4.chat.completions.create(
        #        model="gpt-4o",
        #        messages=[
        #            {"role": "system", "content": prompt},
        #        ]
        #    )

        return prompt

    def _to_probs(self, logits: torch.Tensor, dim=-1):
        max_logits = torch.max(logits, dim=dim, keepdim=True).values
        shifted_logits = logits - max_logits
        exp_logits = torch.exp(shifted_logits)
        probs = exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)
        return probs.tolist()[0]

    def classify(self, text: str):
        encoded_input = self.tokenizer(text, truncation=True, padding=True, max_length=100, return_tensors="pt")
        outputs = self.gpt2(**encoded_input)
        logits = outputs.logits
        return self._to_probs(logits)
    
    def _generate_question(self, prompt, probs: list):
        #print(f'Thinking: {probs[0]*100}%, Feeling: {probs[1]*100}%')
        return self.gpt4.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}] + 
            [{"role": "user", "content": f'Thinking: {probs[0]*100}%, Feeling: {probs[1]*100}%'}],
            temperature=0.7,
            #temperature=0.5,
            top_p=1
        ).choices[0].message.content

    def start(self):
        responses = []
        prompt = self._init_gpt4()
        # First question start with 0% and 0%
        question = self._generate_question(prompt, [0.5,0.5])
        answer = input(f'Question: {question}\n')
        responses.append(question)
        responses.append(answer)
        probs = self.classify('\n'.join(responses))
        #print(probs)
        
        # Second question uses the probs from last question
        question = self._generate_question(prompt, probs)
        answer = input(f'Question: {question}\n')
        responses.append(question)
        responses.append(answer)
        probs = self.classify('\n'.join(responses))
        #print(probs)
        
        # Third question uses the probs from last question
        question = self._generate_question(prompt, probs)
        answer = input(f'Question: {question}\n')
        responses.append(question)
        responses.append(answer)
        probs = self.classify('\n'.join(responses))
        #print(probs)
        
        return 'F' if probs[1] > probs[0] else 'T'