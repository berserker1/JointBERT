import os
import logging
import argparse
from tqdm import tqdm, trange
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from modAL.uncertainty import entropy_sampling
from modAL.models import ActiveLearner

import numpy as np
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                                  intent_label_lst=get_intent_labels(args),
                                                                  slot_label_lst=get_slot_labels(args))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def read_labels(file_name):
    lines = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.read().split('\n')
    if '' in lines:
        lines.remove('')
    return lines


def GP_regression_std(regressor, X):
    # print('Going inside')
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    X = np.asarray(all_input_ids)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return [dataset, X]


def random_forest_active_learner(n_queries, n_initial, X, y, part, incorrect_test_output_values_numbers, predict, total_correct_values):
    initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
    X_training, y_training = X[initial_idx], y[initial_idx]
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    print('Made kernel')
    regressor = ActiveLearner(
        estimator=RandomForestClassifier(),
        query_strategy=entropy_sampling,
        X_training=X_training, y_training=y_training.ravel()
    )
    y_pred = regressor.predict(predict)
    y_pred = y_pred.ravel()
    print('N_queries is', n_queries)
    for idx in range(n_queries):
        query_idx, query_instance = regressor.query(X)
        regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1))

    y_pred_final = regressor.predict(predict)
    y_pred_final = y_pred_final.ravel()
    total_correct_values_numbers = None
    if part != 3:
        print('Random forest regressor score ', regressor.score(predict, incorrect_test_output_values_numbers), 'modAL score function')
    else:
        total_correct_values_numbers = []
        for i in range(len(total_correct_values)):
            total_correct_values_numbers.append(labelled_values[unique.index(total_correct_values[i])])
        print('Random forest regressor score ', regressor.score(predict, total_correct_values_numbers), 'modAL score function')
    return [y_pred_final, total_correct_values_numbers]


def my_stuff(part, pred_config):
    '''Variable declaration here'''

    temp = copy.deepcopy(pred_config)
    args = get_args(pred_config)
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)

    '''final_stuff contains the predictions made by the transformer.'''
    final_stuff = []

    '''File name of the labels of the testing part of the dataset.'''
    test_label_file_name = None

    '''all_test_lines contains input file of the test dataset.'''
    all_test_lines = None

    '''incorrect_test_output contains the input test sample for which the output was wrong.'''
    incorrect_test_output = []

    '''incorrect_test_output_values contains the actual correct outputs.'''
    incorrect_test_output_values = []

    '''indexes_of_wrong_values contains the indexes of input test samples where the output is wrong.'''
    indexes_of_wrong_values = []

    '''correct_test_output contains the input test samples for which the output was correct.'''
    correct_test_output = []

    '''correct_test_output_values contains the corresponding correct outputs.'''
    correct_test_output_values = []

    '''total_correct_values contains the output correct labels'''
    total_correct_values = []

    '''lines variable contains the total samples for training of the active learner,
    it will have different content based upon the 3 cases.'''
    lines = []

    '''y contains all the output labels for training of the active learner,
    it will have different content based upon the 3 cases.'''
    y = []

    '''predict contains the samples to be predicted by the active learner,
    it will have different content based upon the 3 cases.'''
    predict = []

    with open(pred_config.output_file, 'r') as f:
        stuff = f.read().split('\n')
        if '' in stuff:
            stuff.remove('')
        for elem in stuff:
            final_stuff.append(elem.split()[0])
        for i in range(len(final_stuff)):
            final_stuff[i] = final_stuff[i].replace('>', '')
            final_stuff[i] = final_stuff[i].replace('<', '')

    if pred_config.model_dir == 'atis_model':
        temp.input_file = 'data/atis/test/seq.in'
        test_label_file_name = 'data/atis/test/label'
    elif pred_config.model_dir == 'snips_model':
        temp.input_file = 'data/snips/test/seq.in'
        test_label_file_name = 'data/snips/test/label'

    all_test_lines = read_input_file(temp)
    with open(test_label_file_name, 'r') as f:
        stuff = f.read().split('\n')
        if '' in stuff:
            stuff.remove('')
        total_correct_values = stuff
    for i in range(len(stuff)):
        if stuff[i] != final_stuff[i]:
            incorrect_test_output.append(all_test_lines[i])
            incorrect_test_output_values.append(stuff[i])
            indexes_of_wrong_values.append(i)
        else:
            correct_test_output.append(all_test_lines[i])
            correct_test_output_values.append(stuff[i])

    print(len(incorrect_test_output), 'incorrect', len(correct_test_output), 'correct')
    training_file_name = None
    training_labels_file_name = None
    if pred_config.model_dir == 'atis_model':
        training_file_name = 'data/atis/train/seq.in'
        training_labels_file_name = 'data/atis/train/label'
    elif pred_config.model_dir == 'snips_model':
        training_file_name = 'data/snips/train/seq.in'
        training_labels_file_name = 'data/snips/train/label'

    with open(training_file_name, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    y = read_labels(training_labels_file_name)
    c = read_labels(test_label_file_name)
    c = c + y
    predict = incorrect_test_output
    if part == 2:
        lines = lines + correct_test_output
        y = y + correct_test_output_values
    elif part == 3:
        predict = read_input_file(temp)

    unique = list(np.unique(c))

    '''labelled_values is an array used for mapping to converting strings to numbers'''
    labelled_values = []
    for i in range(len(unique)):
        labelled_values.append(i+1)

    '''incorrect_test_output_values_numbers contains the mapped numbers for the incorrect outputs'''
    incorrect_test_output_values_numbers = []

    '''temp variable only'''
    new_y = []

    for elem in y:
        new_y.append(labelled_values[unique.index(elem)])
    y = new_y
    y = np.asarray(y).reshape(-1, 1)
    print(y.shape, 'training labels shape')
    for elem in incorrect_test_output_values:
        incorrect_test_output_values_numbers.append(labelled_values[unique.index(elem)])

    '''Converting training samples to a a numpy matrix'''
    work = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)
    X = work[1]
    print(X.shape, 'training data shape')

    '''Converting the testing samples to a numpy matrix'''
    work = convert_input_file_to_tensor_dataset(predict, pred_config, args, tokenizer, pad_token_label_id)
    predict = work[1]
    print(predict.shape, 'testing data of active learner shape')


    n_initial = 5
    n_queries = 10
    print('N_initial is', n_initial)
    file_name = 'active_learner_modified_' + str(part) + "_" + pred_config.model_dir + '.txt'
    active_learner_output = random_forest_active_learner(n_queries, n_initial, X, y, part,
                                                                      incorrect_test_output_values_numbers, predict, total_correct_values)

    y_pred_final = active_learner_output[0]
    '''temp variable'''
    copy_of_final_stuff = final_stuff.copy()

    active_learner_output_file = 'active_learner_output_' + str(part) + "_" + pred_config.model_dir + '.txt'
    with open(file_name, 'w+') as f:
            new_outputs = []
            if part != 3:
                for i in range(len(indexes_of_wrong_values)):
                    final_stuff[indexes_of_wrong_values[i]] = unique[y_pred_final[i] - 1]
                for elem in final_stuff:
                    val = elem + '\n'
                    new_outputs.append(str(elem))
                    f.write(val)
            else:
                for elem in y_pred_final:
                    val = unique[elem - 1] + '\n'
                    new_outputs.append(unique[elem - 1])
                    f.write(val)
            new_outputs = np.asarray(new_outputs)
            total_correct_values = np.asarray(total_correct_values)
            print((new_outputs == total_correct_values).mean(), 'is the intent acc')

    if part != 3:
        with open(active_learner_output_file, 'w+') as f:
            f.write('Earlier Prediction      ActiveLearner prediction     Correct Value\n')
            for i in range(len(indexes_of_wrong_values)):
                val = str(copy_of_final_stuff[indexes_of_wrong_values[i]]) + '    ' + str(unique[y_pred_final[i] - 1]) + '    ' + str(total_correct_values[indexes_of_wrong_values[i]]) + '\n'
                f.write(val)

    # print(y_pred_final)
    # print(incorrect_test_output_values_numbers)


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    stuff = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)
    dataset = stuff[0]
    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./atis_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    pred_val = None
    try:
        pred_val = int(input('> Do you want to predict and make an output file( press 1 ) or skip ( press 2 )?'))
        if pred_val == 1:
            predict(pred_config)
        elif pred_val == 2:
            pass
        else:
            raise()
    except:
        print('Something wrong in input, exiting!')
        exit()
    value = None
    try:
        value = int(input('> Do you want to run the custom stuff, case 1 or 2 or 3 ?'))
        if value is not None:
            # print('Inside')
            my_stuff(value, pred_config)
    except:
        pass
