# infer_layoutlm_v3

## Information extraction from document images using LayoutLMv3

This plugin proposes inference for document layout analysis using LayoutlMv3. The main task of LayoutLMv3 is extraction of key information from image documents.

Text and visual features are used as inputs. Therefore, it is recommended to use this plugin in a workflow as such: 
1. [infer_mmlab_text_detection](https://github.com/Ikomia-hub/infer_mmlab_text_detection)
2. [infer_mmlab_text_recognition](https://github.com/Ikomia-hub/infer_mmlab_text_recognition)
3. [infer_layoutlm_v3](https://github.com/Ikomia-hub/infer_layoutlm_v3)


The list of available models can be be found on the [Hugging Face model HUB](https://huggingface.co/models?sort=downloads&search=LayoutLMv3)