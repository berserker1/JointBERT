import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }

def modify_data_files(lines, args, task=None):
    if task!= None:
        args.task = task

    training_folder_path = str(args.data_dir).replace('./', '') + '/' + str(args.task) + '/train'
    testing_folder_path = str(args.data_dir).replace('./', '') + '/' + str(args.task) + '/test'
    training_seq_in = training_folder_path + '/seq.in'
    training_seq_out = training_folder_path + '/seq.out'
    training_labels = training_folder_path + '/label'
    testing_seq_in = testing_folder_path + '/seq.in'
    testing_seq_out = testing_folder_path + '/seq.out'
    testing_labels = testing_folder_path + '/label'
    with open(training_seq_in, 'r+') as f, open(training_seq_out, 'r+') as g, open(training_labels, 'r+') as l:
        o_training_seq_in = f.readlines()
        if len(o_training_seq_in) - lines > 0:
            transfer_seq_in = o_training_seq_in[lines:]
            o_training_seq_out = g.readlines()
            transfer_seq_out = o_training_seq_out[lines:]
            o_training_labels = l.readlines()
            transfer_labels = o_training_labels[lines:]
            # Modifying training files
            f.truncate(0)
            g.truncate(0)
            l.truncate(0)
            f.seek(0)
            g.seek(0)
            l.seek(0)
            f.writelines(o_training_seq_in[:lines])
            g.writelines(o_training_seq_out[:lines])
            l.writelines(o_training_labels[:lines])
            print('transfer, ', 'seq.in len ', len(transfer_seq_in), 'seq.out len ', len(transfer_seq_out), 'labels len ', len(transfer_labels))
            print('training orignal, ', 'seq.in len ', len(o_training_seq_in), 'seq.out len ', len(o_training_seq_out), 'labels len ', len(o_training_labels))
        else:
            # Will write the file with its original content
            f.truncate(0)
            f.writelines(o_training_seq_in)
    with open(training_seq_in, 'r') as f, open(training_seq_out, 'r') as g, open(training_labels, 'r') as l:
        print(len(f.readlines()), len(g.readlines()), len(l.readlines()))

    with open(testing_seq_in, 'r') as f, open(testing_seq_out, 'r') as g, open(testing_labels, 'r') as l:
        o_testing_seq_in = f.readlines()
        o_testing_seq_out = g.readlines()
        o_testing_labels = l.readlines()
        print('testing orignal, ', len(o_testing_seq_in), len(o_testing_seq_out), len(o_testing_labels))
    with open(testing_seq_in, 'a+') as f, open(testing_seq_out, 'a+') as g, open(testing_labels, 'a+') as l:
        f.writelines(transfer_seq_in)
        g.writelines(transfer_seq_out)
        l.writelines(transfer_labels)
        f.seek(0)
        g.seek(0)
        l.seek(0)
        print('testing final, ', len(f.readlines()), len(g.readlines()), len(l.readlines()))

    return [training_seq_in, training_seq_out, training_labels, testing_seq_in, testing_seq_out,
            testing_labels, o_training_seq_in, o_training_seq_out, o_training_labels, o_testing_seq_in,
            o_testing_seq_out, o_testing_labels]

def restore_files(parameters, args):
    # Restore the files
    print('Restoring files')
    training_seq_in = parameters[0]
    training_seq_out = parameters[1]
    training_labels = parameters[2]
    testing_seq_in = parameters[3]
    testing_seq_out = parameters[4]
    testing_labels = parameters[5]
    o_training_seq_in = parameters[6]
    o_training_seq_out = parameters[7]
    o_training_labels = parameters[8]
    o_testing_seq_in = parameters[9]
    o_testing_seq_out = parameters[10]
    o_testing_labels = parameters[11]
    with open(training_seq_in, 'w') as f, open(training_seq_out, 'w') as g, open(training_labels, 'w') as l:
        f.writelines(o_training_seq_in)
        g.writelines(o_training_seq_out)
        l.writelines(o_training_labels)
    with open(testing_seq_in, 'w') as f, open(testing_seq_out, 'w') as g, open(testing_labels, 'w') as l:
        f.writelines(o_testing_seq_in)
        g.writelines(o_testing_seq_out)
        l.writelines(o_testing_labels)
