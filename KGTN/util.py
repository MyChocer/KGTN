import json
import numpy as np

NUM_VALID=196+300+311

def process_adjacent_matrix(label_idx_file, testsetup, adjacent_matrix_file, coefficient, process_type, kg_ratio, use_all_base=False):
    '''preprocess the adjacent matrix'''
    # keep novel and base category
    with open(label_idx_file, 'r') as f:
        lowshotmeta = json.load(f)
    novel_classes = lowshotmeta['novel_classes_1']
    novel2_classes = lowshotmeta['novel_classes_2']
    base_classes = lowshotmeta['base_classes_1']
    base2_classes = lowshotmeta['base_classes_2']

    if testsetup:
        if use_all_base:
            ignore_ind = novel_classes 
            valid_nodes = novel2_classes + base2_classes + base_classes
        else:
            ignore_ind = novel_classes + base_classes
            valid_nodes = novel2_classes + base2_classes
    else:
        if use_all_base:
            ignore_ind = novel2_classes
            valid_nodes = novel_classes + base2_classes + base_classes
        else:
            ignore_ind = novel2_classes + base2_classes
            valid_nodes = novel_classes + base_classes

    mat = np.load(adjacent_matrix_file)
    num_classes = mat.shape[0]
    if process_type == 'semantic':

        mat[range(num_classes), range(num_classes)] = 999
        min_mat = np.min(mat, 1) 
        mat = mat - min_mat.reshape(-1, 1) + 1

        in_matrix = coefficient ** (mat - 1)
        in_matrix[:, ignore_ind] = 0
        in_matrix[ignore_ind, :] = 0
        in_matrix[range(num_classes), range(num_classes)] = 2
            
        # in the ascent order
        topk = int(NUM_VALID * kg_ratio / 100)
        max_ = -np.sort(-in_matrix, 1)
        edge = max_[:, topk].reshape(-1, 1)
        in_matrix[in_matrix < edge] = 0
    elif process_type == 'wordnet':

        in_matrix = coefficient ** (mat - 1)
        in_matrix[range(num_classes), range(num_classes)] = 2
        in_matrix[ignore_ind, :] = 0
        in_matrix[:, ignore_ind] = 0
        # in the ascent order
        topk = int(NUM_VALID * kg_ratio / 100)
        max_ = -np.sort(-in_matrix, 1)
        edge = max_[:, topk].reshape(-1, 1)
        in_matrix[in_matrix < edge] = 0
         
    elif process_type == 'mono':
        num_nodes = len(valid_nodes)
        mat = np.ones((num_classes, num_classes)) / num_nodes
        mat[ignore_ind, :] = 0
        mat[:, ignore_ind] = 0
        in_matrix = mat
    elif process_type == 'diagonal':
        mat = np.eye(num_classes) 
        mat[ignore_ind, :] = 0
        mat[:, ignore_ind] = 0
        in_matrix = mat
    elif process_type == 'random':
        mat = np.random.rand(num_classes, num_classes) * 0.2
        mat = np.random.rand(num_classes, num_classes) * 10
        mat[ignore_ind, :] = 0
        mat[:, ignore_ind] = 0
        in_matrix = mat
    return in_matrix


