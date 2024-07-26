from typing import Any
from utils import huggingface_models
from transformers import AutoModel, AutoProcessor, AutoConfig
import resampy
import whisper
import torch
import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__))

# TODO: make this an interface and implement for custom models, including whisper and huggingface

class STModel:

    def __init__(
        self,
        model_key,
        target_language,
        source_language=None,
        additional_args={},
    ) -> None:
        """
        Provides an instance of specified available model (out of Whisper and Huggingface S2TT-transformers), with inference access.
        @param target_language: The target language for the model
        @param source_language: The source language for the model. If not given, autodetected (accents may be a problem)
        @param model_key: Model name, lowercase. Choice of "whisper" or "huggingface"
        @param additional_args: Additional arguments for inference

        To add a custom model, add a new navigation key with custom implemented load and infer methods.
        """
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model_key = model_key
        self.source_language = source_language
        self.target_language = target_language
        # TODO: consider language code mismatch handling
        # self.source_language = language_code_conversion[source_language][model_key]
        # self.target_language = language_code_conversion[target_language][model_key]
        self.model = None
        self.proccessor = None
        self.config = dict()
        self.additional_args = additional_args if additional_args else dict()
        self.navigation = {
            "whisper": {
                "load": lambda: self.load_whisper_model(),
                "infer": lambda audio, sr, rc: self.whisper_infer(audio, sr, return_confidence=rc),
            },
            "huggingface": {
                "load": lambda: self.load_huggingface_model(),
                "infer": lambda audio, sr, rc: self.huggingface_infer(audio, sr, return_confidence=rc),
            }
        }
        print(f'Instace for model key {model_key} created.')
        self.load_model()

    # Model loading
    def load_model(self):
        """
        Load the model specified in the model_key.
        If model key is an integrated model, load via STModel navigation.
        If model key is a Huggingface Key, load via load_huggingface_model.
        If model key is a path to a custom model, load via load_from_path.
        """
        if not self.model_key in self.navigation:
            if self.model_key in huggingface_models:
                self.load_huggingface_model()
                return()
            model_path = self.model_key if os.path.exists(self.model_key) else os.path.join(path, self.model_key)
            assert os.path.exists(model_path), f"Model at {model_path} not found. Please give a valid model path or keyword."
            self.load_from_path(model_path)
        else:
            self.navigation[self.model_key]["load"]()
        print('Model loaded successfully.')

    def load_from_path(model_path):
        """
        Loads custom model loading from local path. Include
        1. A checkpoint file or similar.
        2. An implementation of torch.nn.Module
        3. An inference method logged in self.navigation.
        """
        pass

    def load_whisper_model(self):
        """
        Load the whisper base model. Can be manually changed to other models (choice of tiny, base, small, medium, and large).
        For english transcription, choose {tiny, base, small, medium, large}.en
        """
        self.model = whisper.load_model("base", device=self.device)

    def load_huggingface_model(self):
        """
        Load the Hugging Face model with the specified model_key. Sets the instance model, processor, and config.
        If the model_key is not found in the Hugging Face models, raise an exception.
        """
        assert (
            self.model_key in huggingface_models
        ), f"Model {self.model_key} not found in Hugging Face models"
        self.navigation[self.model_key] = self.navigation["huggingface"]
        self.processor = AutoProcessor.from_pretrained(self.model_key)
        self.model = AutoModel.from_pretrained(self.model_key)
        if self.device == torch.device("cuda"):
            self.model.cuda()
        self.config = AutoConfig.from_pretrained(self.model_key)

    # Inference
    def infer(self, audio_samples, sample_rate=None, return_confidence=False):
        """
        Perform inference on the given audio samples using instance model.
        @param audio_samples: List of audio samples to perform inference on
        @param sample_rate: Sample rate of the audio samples
        @param return_confidence: whether to return the confidence of the model's prediction
        @return dictionary of audio tags and the corresponding model predictions 
        """
        inferenceMethod = self.navigation[self.model_key]["infer"]
        results = dict()
        if return_confidence:
            probabilities = {}
            for tag, sample in audio_samples.items():
                results[tag], probabilities[tag] = inferenceMethod(sample, sample_rate, return_confidence)
            return results, probabilities
        
        for tag, sample in audio_samples.items():
            results[tag] = inferenceMethod(sample, sample_rate, return_confidence)
        return results

    def whisper_infer(self, audio, sample_rate=None, return_confidence=False):
        """
        Perform whisper inference on single audio sample to specified target language. Passes addittional Arguments directly to whisper transcribe.
        TODO: on storage/performance problems bind fp16 to cuda device type (switches to fp16 on gpu, fp32 on cpu)
        @param audio: Audio sample to perform inference on
        @param sample_rate: Sample rate of the audio sample
        @param return_confidence: whether to return the confidence of the model's prediction
        @return: string of textual model prediction based on the audio sample
        """
        whisper_sample_rate = 16000
        if sample_rate != whisper_sample_rate:
            audio = resampy.resample(audio, sample_rate, whisper_sample_rate)
        # whisper inference on audio
        torch.tensor(audio, device=self.device)
        result = self.model.transcribe(
            audio=audio.astype(np.float32),
            language=self.source_language,
            task=(
                "translate"
                if self.source_language != self.target_language
                else "transcribe"
            ),
            fp16=self.device.type == "cuda",
            verbose=False,
            **self.additional_args,
        )
        if not return_confidence:
            return result["text"]
        confidence = np.exp(float(result['segments'][0]['avg_logprob']))
        return result["text"], confidence

    def huggingface_infer(self, audio, sample_rate=None, return_confidence=False):
        """
        Perform Hugging Face inference on single audio sample. Passes addittional Arguments directly to model.generate.
        @param audio: Audio sample to perform inference on
        @param sample_rate: Sample rate of the audio sample
        @param return_confidence: whether to return the confidence of the model's prediction
        @return: string of textual model prediction based on the audio sample
        """
        # TODO: find agnostic variant for AutoLoading, if necessary separate into Language head and non-language head (seamless). How does AutoModel decide which model to load?
        # preprocess audio
        working_audio = resampy.resample(audio, sample_rate, self.model.config.sampling_rate)
        # feed to model
        audio_inputs = self.processor(
            audios=working_audio, sampling_rate=self.model.config.sampling_rate, return_tensors="pt"
        )
        audio_inputs = {k: v.to(device=self.device) for k, v in audio_inputs.items()}
        output_tokens = self.model.generate(**audio_inputs, tgt_lang=self.target_language, generate_speech=False, **self.additional_args)
        translated_text_from_audio = self.processor.decode(
            output_tokens[0].tolist()[0], skip_special_tokens=True
        )
        if not return_confidence:
            return translated_text_from_audio
        return translated_text_from_audio, self.get_prediction_probability(output_tokens)
    
    def get_prediction_probability(self, output):
        """
        Get the prediction probability of the huggingface model's output.
        @param output: The output of the model
        @return: The prediction probability of the output
        """
        transition_scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        probabilities = np.exp(transition_scores.sum(axis=1).cpu())
        return probabilities
    
    def get_token_probabilities(self, output):
        """
        Get the token probabilities of the huggingface model's output.
        @param output: The output of the model
        @return: The token probabilities of the model output as dict, keys being the decoded tokens.
        """
        transition_scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        generated_tokens = output.sequences[:, :]
        token_probs = {}
        for tok, score in zip(generated_tokens[0], transition_scores[0]):
            decoded_token = f"{self.proccessor.decode(tok):8s}"
            token_probs[decoded_token] = np.exp(score.numpy())
        return token_probs
    

    def get_attribute(self, name: str) -> Any:
        """
        Get attribute of the STModel instance. If the attribute is not found, raise an exception.
        @param name: The name of the attribute to get
        """
        attributeDict = {
            "model": self.model,
            "processor": self.processor,
            "config": self.config,
            "model_key": self.model_key,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "additional_args": self.additional_args,
        }
        if name in attributeDict:
            return attributeDict[name]
        else:
            raise Exception(f"STModel has no attribute {name}")
