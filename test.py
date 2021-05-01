import logging
from model import BidirectionalRNN
from utils import createVocabulary, loadVocabulary, validate_model
import os
import argparse


import torch

parser = argparse.ArgumentParser(allow_abbrev=False)

# Network
parser.add_argument("--num_units", type=int, default=64,
                    help="Network size.", dest='layer_size'
                    )

# Training Environment
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size."
                    )

#Model and Vocab
parser.add_argument("--dataset", type=str, default=None,
                    help="""Type 'atis' or 'snips' to use downloaded dataset 
                    or enter what ever you named your own dataset.
                    Note, enter --dataset='' as can not be None"""
                    )
parser.add_argument("--model_path", type=str, default='./model/final.pt',
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

# Print all arguments
for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()

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

slot_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
intent_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

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

checkpoint = torch.load(arg.model_path)
model.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])

valid_slot_f1, valid_intent_accuracy, valid_sem_err, valid_total_loss, \
    valid_slot_loss, valid_intent_loss = validate_model(
        model,
        arg.batch_size,
        os.path.join(full_valid_path, arg.input_file),
        os.path.join(full_valid_path, arg.slot_file),
        os.path.join(full_valid_path, arg.intent_file),
        in_vocab,
        slot_vocab,
        intent_vocab,
        slot_loss_fn,
        intent_loss_fn
    )

logging.info('Valid')
logging.info('Total Loss: ' + str(valid_total_loss))
logging.info('Intent Loss: ' + str(valid_intent_loss))
logging.info('Slot Loss: ' + str(valid_slot_loss))
logging.info('F1 Score: ' + str(valid_slot_f1))
logging.info('Accuracy: ' + str(valid_intent_accuracy))
logging.info('Semantic Err: ' + str(valid_sem_err))
