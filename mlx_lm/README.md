## Generate Text with MLX and :hugs: Hugging Face

This an example of large language model text generation that can pull models from
the Hugging Face Hub.

For more information on this example, see the [README](../README.md) in the
parent directory.

This package also supports fine tuning with LoRA or QLoRA. For more information
see the [LoRA documentation](LORA.md).

### Validate native MTP penalty parity

For a local MTP checkpoint, the validation helper compares greedy vanilla and
native-MTP tokens with the requested penalty processors. It does not require a
model download:

```bash
python -m mlx_lm.examples.validate_mtp_penalties \
  --model /path/to/local-mtp-model --max-tokens 64 --mtp-depth 4 \
  --presence-penalty 1.5 --repetition-penalty 1.1 --context-size 64 \
  --compare-logprobs
```
