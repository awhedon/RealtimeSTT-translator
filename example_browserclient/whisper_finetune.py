import argparse
import os
import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from audiomentations import Compose, TimeStretch, Gain, PitchShift, OneOf, AddBackgroundNoise, AddGaussianNoise, PolarityInversion
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from google.cloud import speech
import io

AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"

def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    if audio_column_name is not None and audio_column_name != AUDIO_COLUMN_NAME:
        ds = ds.rename_column(audio_column_name, AUDIO_COLUMN_NAME)
    if text_column_name is not None and text_column_name != TEXT_COLUMN_NAME:
        ds = ds.rename_column(text_column_name, TEXT_COLUMN_NAME)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.remove_columns(set(ds.features.keys()) - set([AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME]))
    return ds

def load_datasets(language):
    raw_datasets = DatasetDict()

    if language == "en":
        ds_train = load_dataset("librispeech_asr", "clean", split="train.100")
        ds_eval = load_dataset("librispeech_asr", "clean", split="test")
    elif language == "es":
        ds_train = load_dataset("mozilla-foundation/common_voice_11_0", "es", split="train")
        ds_eval = load_dataset("mozilla-foundation/common_voice_11_0", "es", split="test")
    else:
        raise ValueError("Unsupported language. Choose 'en' or 'es'.")

    raw_datasets["train"] = normalize_dataset(ds_train, text_column_name="text")
    raw_datasets["eval"] = normalize_dataset(ds_eval, text_column_name="text")

    return raw_datasets

def setup_augmentation(musan_dir):
    return Compose([
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.2, leave_length_unchanged=False),
        Gain(min_gain_in_db=-6, max_gain_in_db=6, p=0.1),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.2),
        OneOf([
            AddBackgroundNoise(sounds_path=musan_dir, min_snr_in_db=1.0, max_snr_in_db=5.0, noise_transform=PolarityInversion(), p=1.0),
            AddGaussianNoise(min_amplitude=0.005, max_amplitude=0.015, p=1.0),
        ], p=0.2),
    ])

def augment_dataset(batch, augmentation):
    sample = batch[AUDIO_COLUMN_NAME]
    augmented_waveform = augmentation(sample["array"], sample_rate=sample["sampling_rate"])
    batch[AUDIO_COLUMN_NAME]["array"] = augmented_waveform
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def transcribe_google(audio_content, language_code):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
    )

    response = client.recognize(config=config, audio=audio)
    return " ".join(result.alternatives[0].transcript for result in response.results)

def evaluate_wer(model, processor, dataset, normalizer, do_normalize_eval, device, use_google_api=False, language_code=None):
    metric = evaluate.load("wer")
    
    def map_to_pred(batch):
        audio = batch["audio"]
        if use_google_api:
            audio_content = io.BytesIO(audio["array"].tobytes())
            transcription = transcribe_google(audio_content.getvalue(), language_code)
        else:
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = model.generate(input_features.to(device))[0]
            transcription = processor.decode(predicted_ids)
        batch["prediction"] = transcription
        return batch

    result = dataset.map(map_to_pred, remove_columns=dataset.column_names)
    
    predictions = result["prediction"]
    references = result["sentence"]

    if do_normalize_eval:
        predictions = [normalizer(pred) for pred in predictions]
        references = [normalizer(ref) for ref in references]

    wer = metric.compute(predictions=predictions, references=references)
    return wer

def compute_metrics(pred, tokenizer, normalizer, do_normalize_eval):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    if do_normalize_eval:
        pred_str = [normalizer(pred) for pred in pred_str]
        label_str = [normalizer(label) for label in label_str]

    wer = evaluate.load("wer").compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def main(args):
    # Set up GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{args.model_size}")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model = model.to(device)

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{args.model_size}", language=args.language, task="transcribe")

    # Load and prepare datasets
    raw_datasets = load_datasets(args.language)
    
    augmentation = setup_augmentation(args.musan_dir)
    augmented_raw_training_dataset = raw_datasets["train"].map(
        lambda batch: augment_dataset(batch, augmentation),
        num_proc=args.preprocessing_num_workers,
        desc="augment train dataset"
    )

    raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], augmented_raw_training_dataset])
    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=10)

    normalizer = BasicTextNormalizer()

    def prepare_dataset(batch):
        audio = batch[AUDIO_COLUMN_NAME]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        input_str = normalizer(batch[TEXT_COLUMN_NAME]).strip() if args.do_normalize_text else batch[TEXT_COLUMN_NAME]
        batch["labels"] = processor.tokenizer(input_str).input_ids
        return batch

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        num_proc=args.preprocessing_num_workers,
        remove_columns=next(iter(raw_datasets.values())).column_names,
        desc="preprocess dataset",
    )

    def is_audio_in_length_range(length):
        return args.min_input_length < length < args.max_input_length

    vectorized_datasets = vectorized_datasets.filter(
        is_audio_in_length_range,
        num_proc=args.preprocessing_num_workers,
        input_columns=["input_length"]
    )

    def is_labels_in_length_range(labels):
        return len(labels) < model.config.max_length

    vectorized_datasets = vectorized_datasets.filter(
        is_labels_in_length_range,
        num_proc=args.preprocessing_num_workers,
        input_columns=["labels"]
    )

    # Setup training
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"],
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["eval"],
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer, normalizer, args.do_normalize_eval),
    )

    # Train and save
    trainer.train()
    model.save_pretrained(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    # Evaluate original Whisper, fine-tuned Whisper, and Google API
    original_model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{args.model_size}").to(device)
    
    print("Evaluating original Whisper model...")
    original_wer = evaluate_wer(original_model, processor, raw_datasets["eval"], normalizer, args.do_normalize_eval, device)
    print(f"Original Whisper model WER: {original_wer}")

    print("Evaluating fine-tuned Whisper model...")
    finetuned_wer = evaluate_wer(model, processor, raw_datasets["eval"], normalizer, args.do_normalize_eval, device)
    print(f"Fine-tuned Whisper model WER: {finetuned_wer}")

    if args.use_google_api:
        print("Evaluating Google Speech-to-Text API...")
        language_code = "en-US" if args.language == "en" else "es-ES"
        google_wer = evaluate_wer(None, None, raw_datasets["eval"], normalizer, args.do_normalize_eval, device, use_google_api=True, language_code=language_code)
        print(f"Google Speech-to-Text API WER: {google_wer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for ASR")
    parser.add_argument("--language", type=str, choices=["en", "es"], required=True, help="Language to fine-tune on (en or es)")
    parser.add_argument("--model_size", type=str, default="medium", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--musan_dir", type=str, default="./musan", help="Path to MUSAN dataset for noise augmentation")
    parser.add_argument("--preprocessing_num_workers", type=int, default=4, help="Number of workers for preprocessing")
    parser.add_argument("--do_normalize_text", type=bool, default=True, help="Whether to normalize text")
    parser.add_argument("--do_normalize_eval", type=bool, default=True, help="Whether to normalize evaluation text")
    parser.add_argument("--min_input_length", type=float, default=0, help="Minimum input length in seconds")
    parser.add_argument("--max_input_length", type=float, default=30, help="Maximum input length in seconds")
    parser.add_argument("--output_dir", type=str, default="./outputs/whisper_ft", help="Output directory")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Per-device training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Per-device evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=4000, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Whether to use gradient checkpointing")
    parser.add_argument("--fp16", type=bool, default=True, help="Whether to use fp16 precision")
    parser.add_argument("--use_google_api", action="store_true", help="Whether to evaluate using Google Speech-to-Text API")

    args = parser.parse_args()
    main(args)