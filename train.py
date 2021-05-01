from utils import conv_to_tensor, calculate_loss, log_in_tensorboard
from utils import createVocabulary, loadVocabulary, validate_model
from utils import DataProcessor, calculate_metrics, create_f1_lists
from model import BidirectionalRNN
import os
import argparse


import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


parser = argparse.ArgumentParser(allow_abbrev=False)

# Network
parser.add_argument("--num_units", type=int, default=64,
                    help="Network size.", dest='layer_size'
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
                    help="Disable early stop, which is based on"
                    + " sentence level accuracy."
                    )
parser.add_argument("--patience", type=int, default=5,
                    help="Patience to wait before stop."
                    )

#Model and Vocab
parser.add_argument("--dataset", type=str, default=None,
                    help="""Type 'atis' or 'snips' to use downloaded dataset 
                    or enter what ever you named your own dataset.
                    Note, enter --dataset='' as can not be None"""
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

# Print all arguments
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

# load dataset
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


# training variable declarations
epochs = 0
epoch_loss = 0.0
epoch_slot_loss = 0.0
epoch_intent_loss = 0.0
data_processor = None
steps_in_epoch = 0
total_steps = 0
no_improve = 0

pred_intents = []
correct_intents = []
slot_outputs_pred = []
correct_slots = []
input_words = []

tb_log_writer = SummaryWriter()

# used for early stopping
valid_acc = 0

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
slot_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
intent_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")


# training loop
while True:

    # get data
    if data_processor is None:
        data_processor = DataProcessor(
            os.path.join(full_train_path, arg.input_file),
            os.path.join(full_train_path, arg.slot_file),
            os.path.join(full_train_path, arg.intent_file),
            in_vocab, slot_vocab, intent_vocab
        )

    input_data, slots, slot_weights, seq_length, intent, _, _, _ \
        = data_processor.get_batch(arg.batch_size)

    input_data, slots, slot_weights, seq_length, intent \
        = conv_to_tensor(input_data, slots, slot_weights, seq_length, intent)

    # model predict
    slot_outputs, intent_output = model.forward(input_data=input_data)

    # loss
    slot_loss, intent_loss, total_loss = calculate_loss(
        slots, slot_outputs, slot_weights, slot_loss_fn,
        intent_output, intent, intent_loss_fn, arg.batch_size
    )

    # f1 metrics
    p_i, c_i, s_o, c_o, i_w = create_f1_lists(
        slot_outputs, intent_output, intent, slots,
        input_data, seq_length, slot_vocab, in_vocab
    )
    pred_intents.extend(p_i)
    correct_intents.extend(c_i)
    slot_outputs_pred.extend(s_o)
    correct_slots.extend(c_o)
    input_words.extend(i_w)

    # backprop
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    total_steps += 1
    steps_in_epoch += 1
    epoch_loss += total_loss
    epoch_slot_loss += slot_loss
    epoch_intent_loss += intent_loss

    # if epoch is finished
    if data_processor.end == 1:

        epochs += 1

        # clean up data_processor
        data_processor.close()
        data_processor = None

        # calculate train metrics
        f1, precision, recall, accuracy, semantic_acc = calculate_metrics(
            pred_intents, correct_intents, slot_outputs_pred, correct_slots
        )

        log_in_tensorboard(
            tb_log_writer, epochs, "train",
            epoch_loss/steps_in_epoch, epoch_intent_loss/steps_in_epoch,
            epoch_intent_loss/steps_in_epoch, f1, accuracy, semantic_acc
        )

        # reset steps and epoch loss
        steps_in_epoch = 0
        epoch_loss = 0.0
        epoch_slot_loss = 0.0
        epoch_intent_loss = 0.0
        # clean up epoch variables
        pred_intents = []
        correct_intents = []
        slot_outputs_pred = []
        correct_slots = []
        input_words = []

        # save model
        save_path = os.path.join(
            arg.model_path, 'epoch_' + str(epochs) + '.pt'
        )
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': epoch_loss,
        }, save_path)

        # validation
        valid_slot_f1, valid_intent_accuracy, valid_sem_acc,\
            valid_total_loss, valid_slot_loss, valid_intent_loss \
            = validate_model(
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
        log_in_tensorboard(
            tb_log_writer, epochs, "valid",
            valid_total_loss, valid_intent_loss, valid_slot_loss,
            valid_slot_f1, valid_intent_accuracy, valid_sem_acc
        )

        # test set
        test_slot_f1, test_intent_accuracy, test_sem_acc,\
            test_total_loss, test_slot_loss, test_intent_loss \
            = validate_model(
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
        log_in_tensorboard(
            tb_log_writer, epochs, "test",
            test_total_loss, test_intent_loss, test_slot_loss,
            test_slot_f1, test_intent_accuracy, test_sem_acc
        )

        if test_sem_acc <= valid_acc:
            no_improve += 1
        else:
            valid_acc = test_sem_acc
            no_improve = 0

        if epochs == arg.max_epochs:
            break

        if arg.early_stop == True:
            if no_improve > arg.patience:
                break

tb_log_writer.close()

torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'loss': epoch_loss,
}, "./model/final.pt")
