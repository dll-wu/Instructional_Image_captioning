# Instructional Fine-tuning for Image Captioning Models
This repository contains the official code and models for the submission titled "Training Captioning Models to Follow Instructions with Human-like Supervision" for ICASSP 2024.

## Abstract
Instructional fine-tuning with an emphasis on human alignment has demonstrated advantages for conditional text generation. However, achieving similar alignment for image captioning remains elusive. Traditional image captioning models primarily emphasize end-to-end training, often leading to oversimplified captions that do not capture the comprehensive essence of visual content. In contrast, state-of-the-art (SOTA) methods rely heavily on prealigned large language models (LLM) but seldom explore standalone instructional alignment.

To bridge this gap, we study aligning predecessors of image captioning models with humans by leveraging supervision from well-aligned LLMs. Specifically, we employ ChatGPT to summarize and paraphrase diverse image captions into coherent paragraphs, as human-like instructional supervision. We then fine-tune the captioning model using the collected instructional dataset. Empirical evaluations on the COCO Captions dataset showcase remarkable enhancements in caption quality with our approach.

Find our implementation and trained models in this repository.

## Installation

Before running the code, you need to install the required packages:

```bash
pip install peft bitsandbytes accelerate
```

## Models

We provide four trained models for our approach:

- [git-base-refines](https://huggingface.co/kurileo/git-base-refines)
- [git-large-refines](https://huggingface.co/kurileo/git-large-refines)
- [blip2-opt-2.7b-refines](https://huggingface.co/kurileo/blip2-opt-2.7b-refines)
- [blip2-opt-6.7b-refines](https://huggingface.co/kurileo/blip2-opt-6.7b-refines)

## Usage

### Inference
`python demo/run_git.py` or `python demo/run_blip2.py`

*Note:* fine-tuned blip2 models will generate illegal characters, post-filtering is needed for any practical usage.

### Training

*Will be updated later*

## Citation

If you use our code, models, or findings in your research, please cite our work (you'll want to add the BibTeX entry once your paper is published):

```bibtex
TO be updated
```

## Contact

For any questions or concerns, please open an issue in this repository or contact [me](mailto:yanyang@westlake.edu.cn).

## License

This repository is licensed under the WTFPL license. Enjoy!
