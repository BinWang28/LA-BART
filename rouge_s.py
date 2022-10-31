#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Author: Bin Wang
# -----
# Copyright (c) 2022 National University of Singapore
# -----

import rouge
import logging

def prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def py_rouge_scores(generated, reference, scores=None):
    '''
        load and display scores
    '''

    if not scores:

        apply_avg = True
        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                            max_n=2,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=apply_avg,
                            apply_best=False,
                            alpha=0.5, # Default F1_score
                            weight_factor=1.2,
                            stemming=True)
        all_hypothesis = generated
        all_references = reference
        scores = evaluator.get_scores(all_hypothesis, all_references)
        
        logging.info("")
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            logging.info(prepare_results(metric, results['p'], results['r'], results['f']))
        logging.info("")

        return scores

    else:

        logging.info("")
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            logging.info(prepare_results(metric, results['p'], results['r'], results['f']))
        logging.info("")

    

