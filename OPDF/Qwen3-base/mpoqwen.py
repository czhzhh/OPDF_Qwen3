from transformers import Qwen3Model, Qwen3PreTrainedModel, Qwen3Config
import torch.nn as nn

class QwenModelCustom(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=True,
            use_cache=False,
        )

        # Qwen3.1 has no dedicated pooler, so reuse the first token representation.
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0, :]

        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits, outputs.hidden_states

        return logits, outputs.hidden_states
