"""Module of class model for RoBERTa classifier used for Wikidata linking classification."""
from typing import Any, Optional, Sequence

import torch
import torch.multiprocessing

# from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer, RobertaForSequenceClassification, RobertaTokenizerFast


class WikidataLinkingClassifier(torch.nn.Module):
    """Pytorch Wikidata linking classifier class."""

    def __init__(  # type: ignore
        self,
        model_type: str = "roberta-large-mnli",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        dropout_p: float = 0.1,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        epochs: int = 10,
        steps_per_epoch: int = 1000,
        train_dataloaders=None,
    ) -> None:
        """Init function."""
        super().__init__()
        self.model_type = model_type
        if tokenizer is None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(model_type)
        else:
            self.tokenizer = tokenizer
        self.model = RobertaForSequenceClassification.from_pretrained(model_type)
        self.model.config.hidden_dropout_prob = dropout_p
        self.model.config.attention_probs_dropout_prob = dropout_p
        self.lr = lr  # pylint: disable=invalid-name
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_dataloaders = train_dataloaders

    def forward(  # type: ignore # pylint: disable=arguments-differ
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ):
        """Function for performing forward pass through network.

        Returns:
            A dictionary containing loss values, logits, and other information.
        """
        return self.model(**x, labels=y)

    def infer(self, text: str, qnode_texts: Sequence[str]) -> Any:
        """A function to take two string inputs, format them, and generate predictions for them.

        Returns:
            A dictionary containing loss values, logits, and other information.
        """
        paired_text = [(text, qnode_text) for qnode_text in qnode_texts]
        encoded_input = self.tokenizer.batch_encode_plus(
            paired_text,
            pad_to_max_length=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            return self.forward(encoded_input)
