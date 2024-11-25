from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
import torch
import os

class SnClassifier:
    '''
    Prerequisite:
    1. A `tokenizer` folder under the same directory to be loaded by `AutoTokenizer`, it contains:
       - `added_tokens.json`
       - `merges.txt`
       - `special_tokens_map.json`
       - `tokenizer_config.json`
       - `tokenizer.json`
       - `vocab.json`
    2. A `model` folder under the same directory to be loaded by `AutoModelForSequenceClassification`,
       it contains:
       - `config.json`
       - `model.safetensors`.
    3. `OPENAI_API_KEY` set in `os.environ`

    Example:
    from sn_agent import SnClassifier
    classifier = SnClassifier()
    classifier.start()
    >> 'N'
    '''

    def __init__(self):
        self.labels = {0: "N", 1: "S"}
        self.tokenizer = None
        self.gpt2 = None
        self.gpt4 = OpenAI()
        
        self._load_tokenizer()
        self._load_gpt2()

    def _load_tokenizer(self):
        tokenizer_path = "tokenizer"
        assert(os.path.exists(tokenizer_path))

        print("Loading existing tokenizer from:", tokenizer_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def _load_gpt2(self):
        model_path = 'model'
        assert(os.path.exists(model_path))

        print("Loading existing model from:", model_path)
        self.gpt2 = AutoModelForSequenceClassification.from_pretrained(model_path)

    def _init_gpt4(self):
        prompt_path = 'prompt.txt'
        assert(['OPENAI_API_KEY'] in os.environ)
        assert(os.path.exists(prompt_path))

        with open(prompt_path, 'r') as file:
            prompt = file.read()
            _ = self.gpt4.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                ]
            )

    def _to_probs(self, logits: torch.Tensor, dim=-1):
        '''
        Convert logits to binary probabilities.

        Example:
        _to_probs(tensor.Torch([[-0.8, 0.2]]))
        >> [0.15, 0.85]
        '''
        max_logits = torch.max(logits, dim=dim, keepdim=True).values
        shifted_logits = logits - max_logits
        exp_logits = torch.exp(shifted_logits)
        probs = exp_logits / torch.sum(exp_logits, dim=dim, keepdim=True)
        return probs.tolist()[0]

    def classify(self, text: str):
        '''
        Classify a text to either N or S.

        Example:
        classify('random text')
        >> [0.3, 0.7]
        '''
        encoded_input = self.tokenizer(text, truncation=True, padding=True, max_length=100, return_tensors="pt")
        outputs = self.gpt2(**encoded_input)
        logits = outputs.logits
        return self._to_probs(logits)
    
    def _generate_question(self, probs: list):
        return self.gpt4.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f'Sensing: {probs[0]*100}%, Natural: {probs[1]*100}%'},
            ],
            temperature=0.7,
            top_p=1
        ).choices[0].message.content

    def start(self):
        responses = []
        self._init_gpt4()
        # First question start with 0% and 0%
        question = self._generate_question([0,0])
        answer = input(f'Question: {question}\n')
        responses.append(question)
        responses.append(answer)
        probs = self.classify('\n'.join(responses))

        # Second question uses the probs from last question
        question = self._generate_question(probs)
        answer = input(f'Question: {question}\n')
        responses.append(question)
        responses.append(answer)
        probs = self.classify('\n'.join(responses))

        # Third question uses the probs from last question
        question = self._generate_question(probs)
        answer = input(f'Question: {question}\n')
        responses.append(question)
        responses.append(answer)
        probs = self.classify('\n'.join(responses))

        return 'N' if probs[0] > probs[1] else 'S'
