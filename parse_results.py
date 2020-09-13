# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import json
import argparse
import numpy as np
import os
def parse_args():
    parser = argparse.ArgumentParser(description='parse_results')
    parser.add_argument('--resultsdir', type=str)
    parser.add_argument('--repr', type=str)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=0.001, type=float)
    parser.add_argument('--max_per_label', default=0, type=int)
    parser.add_argument('--matchingnetwork', default=0,type=int)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    if args.matchingnetwork:
        outpath='MN_{}_expid_{:d}_lowshotn_{:d}.json'
    else:
        outpath = '{}_' + 'lr_{:.3f}_wd_{:.3f}'.format(args.lr, args.wd) + '_expid_{:d}_lowshotn_{:d}.json' 
    lowshotns = [1,2,5, 10]
    expids = [1,2,3,4,5]
    all_nums = np.zeros((len(lowshotns), len(expids), 4))
    for i, ei in enumerate(expids):
        for j, ln in enumerate(lowshotns):
            outfile = os.path.join(args.resultsdir, outpath.format(args.repr, ei, ln))
            with open(outfile, 'r') as f:
                out = json.load(f)
                all_nums[j,i,:] = np.array(out['accs'])*100

    means = np.mean(all_nums, axis=1)
    print('n \tNovel17 Novel18 All    All with prior')
    print('  \tTop-5   Top-5   Top-5  Top-5')
    
    print('='*80)

    to_print = '\n'.join([str(lowshotns[i])+'\t'+'\t'.join(['{:.2f}'.format(x) for x in y]) for i, y in enumerate(means)])
    print(to_print)
    print('='*80)
    mean_mean = np.mean(means, axis=0)
    print('mean\t'+'\t'.join(['{:.2f}'.format(x) for x in mean_mean]))


