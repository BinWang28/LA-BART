# A Focused Study on Sequence Length for Dialogue Summarization

For more details, please find our paper on arXiv

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.11910)

# Requirements

We test the code with python 3.7 and below requirements.
```
pip install -r requirements.txt
```

# Training and Testing

Baseline model (BART-Large) on DialogSum dataset
```
bash 1_dialogsum_bart_large.sh
```

Baseline model (BART-Large) on SAMSum dataset
```
bash 2_samsum_bart_large.sh
```

LA-BART model (BART-Large) on DialogSum dataset
```
bash 3_dialogsum_la_bart_large.sh
```

LA-BART model (BART-Large) on SAMSum dataset
```
bash 4_samsum_la_bart_large.sh
```

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