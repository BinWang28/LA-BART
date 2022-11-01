# A Focused Study on Sequence Length for Dialogue Summarization

For more details, please find our paper on arXiv

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.11910)

# Requirements

We test the code with python 3.7 and below requirements.
```
pip install -r requirements.txt
```

# Training and Testing

## Baseline model (BART-Large) on DialogSum dataset
```
bash 1_dialogsum_bart_large.sh
```
**ROUGE-1: 47.36, ROUGE-2: 21.23, ROUGE-L: 44.88**

## Baseline model (BART-Large) on SAMSum dataset
```
bash 2_samsum_bart_large.sh
```
**ROUGE-1: 52.31, ROUGE-2: 27.57, ROUGE-L: 49.57**

## LA-BART model (BART-Large) on DialogSum dataset
```
bash 3_dialogsum_la_bart_large.sh
```
**ROUGE-1: 49.42, ROUGE-2: 22.37, ROUGE-L: 46.93**

## LA-BART model (BART-Large) on SAMSum dataset
```
bash 4_samsum_la_bart_large.sh
```
**ROUGE-1: 57.44, ROUGE-2: 30.96, ROUGE-L: 53.23**

The above hyperparameters are not carefully tuned. We can still see the effect of controlling the output length.

## References

If you find our work useful, please consider citing our work.

```
@article{wang2022focused,
  title={A Focused Study on Sequence Length for Dialogue Summarization},
  author={Wang, Bin and Zhang, Chen and Wei, Chengwei and Li, Haizhou},
  journal={arXiv preprint arXiv:2209.11910},
  year={2022}
}
```

```
@article{to update with proceedings,
  title={==},
  author={==},
  journal={==},
  year={==}
}
```

Contact to Bin Wang at [bwang28c@gmail.com](mailto:bwang28c@gmail.com) for any issues.