import os
import argparse
import numpy as np

import torch

import logging

from model import BidirectionalRNN

from utils import createVocabulary
from utils import loadVocabulary
from utils import computeF1Score
from utils import DataProcessor

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(allow_abbrev=False)

# Network
parser.add_argument("--num_units", type=int, default=64,
                    help="Network size.", dest='layer_size'
                    )
parser.add_argument("--model_type", type=str, default='full',
                    help="""full(default) | intent_only
            full: full attention model
            intent_only: intent attention model"""
                    )

# Training Environment
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size."
                    )
parser.add_argument("--max_epochs", type=int, default=20,
                    help="Max epochs to train."
                    )
parser.add_argument("--no_early_stop", action='store_false',
                    dest='early_stop',
                    help="Disable early stop, which is based on sentence level accuracy."
                    )
parser.add_argument("--patience", type=int, default=5,
                    help="Patience to wait before stop."
                    )

#Model and Vocab
parser.add_argument("--dataset", type=str, default=None,
                    help="""Type 'atis' or 'snips' to use dataset provided by us 
            or enter what ever you named your own dataset.
            Note, if you don't want to use this part, enter --dataset=''. 
            It can not be None"""
                    )
parser.add_argument("--model_path", type=str, default='./model',
                    help="Path to save model."
                    )
parser.add_argument("--vocab_path", type=str, default='./vocab',
                    help="Path to vocabulary files."
                    )

# Data
parser.add_argument("--train_data_path", type=str, default='train',
                    help="Path to training data files."
                    )
parser.add_argument("--test_data_path", type=str, default='test',
                    help="Path to testing data files."
                    )
parser.add_argument("--valid_data_path", type=str, default='valid',
                    help="Path to validation data files."
                    )
parser.add_argument("--input_file", type=str, default='seq.in',
                    help="Input file name."
                    )
parser.add_argument("--slot_file", type=str, default='seq.out',
                    help="Slot file name."
                    )
parser.add_argument("--intent_file", type=str, default='label',
                    help="Intent file name."
                    )

arg = parser.parse_args()

# Print arguments
for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()

# full path to data will be: ./data + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'snips':
    print('use snips dataset')
elif arg.dataset == 'atis':
    print('use atis dataset')
else:
    print('use own dataset: ', arg.dataset)
full_train_path = os.path.join('./data', arg.dataset, arg.train_data_path)
full_test_path = os.path.join('./data', arg.dataset, arg.test_data_path)
full_valid_path = os.path.join('./data', arg.dataset, arg.valid_data_path)

createVocabulary(
    os.path.join(full_train_path, arg.input_file),
    os.path.join(arg.vocab_path, 'in_vocab')
)
createVocabulary(
    os.path.join(full_train_path, arg.slot_file),
    os.path.join(arg.vocab_path, 'slot_vocab')
)
createVocabulary(
    os.path.join(full_train_path, arg.intent_file),
    os.path.join(arg.vocab_path, 'intent_vocab')
)

in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))

epochs = 0
loss = 0.0
data_processor = None
line = 0
num_loss = 0
step = 0
no_improve = 0

# variables to store highest values among epochs, only use 'valid_err' for now
valid_slot = 0
test_slot = 0
valid_intent = 0
test_intent = 0
valid_err = 0
test_err = 0

model = BidirectionalRNN(
    input_size=len(in_vocab['vocab']),
    sequence_length=arg.batch_size,
    slot_size=len(slot_vocab['vocab']),
    intent_size=len(intent_vocab['vocab']),
    layer_size=arg.layer_size,
    isTraining=True,
    remove_slot_attn=False,
    add_final_state_to_intent=True
)

learning_rate = 1e-3
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

while True:
    if data_processor == None:
        data_processor = DataProcessor(
            os.path.join(full_train_path, arg.input_file),
            os.path.join(full_train_path, arg.slot_file),
            os.path.join(full_train_path, arg.intent_file),
            in_vocab, slot_vocab, intent_vocab
        )
    input_data, slots, slot_weights, sequence_length, intent, _, _, _ \
        = data_processor.get_batch(arg.batch_size)

    input_data = torch.tensor(input_data)
    slots = torch.tensor(slots)
    slot_weights = torch.tensor(slot_weights)
    sequence_length = torch.tensor(sequence_length)
    intent = torch.tensor(intent)

    slots_shape = slots.size()
    slots_reshape = slots.reshape([-1])

    training_outputs = model.forward(input_data=input_data)

    # slot loss
    slot_outputs = training_outputs[0]
    slot_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    crossent = slot_loss_fn(slot_outputs, slots_reshape.long())
    crossent = crossent.reshape(slots_shape)
    slot_loss = torch.sum(crossent * slot_weights, 1)
    total_size = torch.sum(slot_weights, 1)
    total_size += 1e-12
    slot_loss = torch.sum(slot_loss / total_size)

    # intent loss
    intent_output = training_outputs[1]
    intent_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    crossent = intent_loss_fn(intent_output, intent.long())
    intent_loss = torch.sum(crossent) / arg.batch_size

    total_loss = slot_loss + intent_loss

    optim.zero_grad()

    total_loss.backward()

    optim.step()

    num_loss += 1
    loss += total_loss

    if data_processor.end == 1:
        line = 0
        data_processor.close()
        data_processor = None
        epochs += 1
        # logging.info('Step: ' + str(step))
        logging.info('Epochs: ' + str(epochs))
        logging.info('Loss: ' + str(loss/num_loss))
        num_loss = 0
        loss = 0.0

        save_path = os.path.join(
            arg.model_path, 'epoch_' + str(epochs) + '.pt'
        )
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
        }, save_path)

        def valid(in_path, slot_path, intent_path):
            data_processor_valid = DataProcessor(
                in_path, slot_path, intent_path, 
                in_vocab, slot_vocab, intent_vocab
            )

            pred_intents = []
            correct_intents = []
            slot_outputs = []
            correct_slots = []
            input_words = []

            # used to gate
            gate_seq = []
            while True:
                in_data, slot_data, _, length, intents, _, _, _ \
                    = data_processor_valid.get_batch(arg.batch_size)

                in_data = torch.tensor(in_data)
                slot_out, intent_out = model.forward(input_data=in_data)

                slot_out = slot_out.cpu().detach().numpy()
                intent_out = intent_out.cpu().detach().numpy()

                for i in intent_out:
                    pred_intents.append(np.argmax(i))
                for i in intents:
                    correct_intents.append(i)

                pred_slots = slot_out.reshape(
                    (slot_data.shape[0], slot_data.shape[1], -1))
                for p, t, i, l in zip(pred_slots, slot_data, in_data, length):
                    p = np.argmax(p, 1)
                    tmp_pred = []
                    tmp_correct = []
                    tmp_input = []
                    for j in range(l):
                        tmp_pred.append(slot_vocab['rev'][p[j]])
                        tmp_correct.append(slot_vocab['rev'][t[j]])
                        tmp_input.append(in_vocab['rev'][i[j]])

                    slot_outputs.append(tmp_pred)
                    correct_slots.append(tmp_correct)
                    input_words.append(tmp_input)

                if data_processor_valid.end == 1:
                    break

            pred_intents = np.array(pred_intents)
            correct_intents = np.array(correct_intents)
            accuracy = (pred_intents == correct_intents)
            semantic_error = accuracy
            accuracy = accuracy.astype(float)
            accuracy = np.mean(accuracy)*100.0

            index = 0
            for t, p in zip(correct_slots, slot_outputs):
                # Process Semantic Error
                if len(t) != len(p):
                    raise ValueError('Error!!')

                for j in range(len(t)):
                    if p[j] != t[j]:
                        semantic_error[index] = False
                        break
                index += 1
            semantic_error = semantic_error.astype(float)
            semantic_error = np.mean(semantic_error)*100.0

            f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
            logging.info('slot f1: ' + str(f1))
            logging.info('intent accuracy: ' + str(accuracy))
            logging.info(
                'semantic error(intent, slots are all correct): ' \
                + str(semantic_error)
            )

            data_processor_valid.close()
            return f1, accuracy, semantic_error, pred_intents, \
                correct_intents, slot_outputs, correct_slots, \
                input_words, gate_seq

        logging.info('Valid:')
        epoch_valid_slot, epoch_valid_intent, epoch_valid_err, \
            valid_pred_intent, valid_correct_intent, valid_pred_slot, \
            valid_correct_slot, valid_words, valid_gate \
            = valid(
                os.path.join(full_valid_path, arg.input_file), 
                os.path.join(full_valid_path, arg.slot_file), 
                os.path.join(full_valid_path, arg.intent_file)
            )

        logging.info('Test:')
        epoch_test_slot, epoch_test_intent, epoch_test_err, test_pred_intent, \
            test_correct_intent, test_pred_slot, test_correct_slot, \
            test_words, test_gate = valid(
                os.path.join(full_test_path, arg.input_file), 
                os.path.join(full_test_path, arg.slot_file), 
                os.path.join(full_test_path, arg.intent_file)
            )

        if epoch_valid_err <= valid_err:
            no_improve += 1
        else:
            valid_err = epoch_valid_err
            no_improve = 0

        if epochs == arg.max_epochs:
            break

        if arg.early_stop == True:
            if no_improve > arg.patience:
                break

torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': loss,
}, "./model/final.pt")