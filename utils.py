import logging
import numpy as np
import torch

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)


def createVocabulary(input_path, output_path, no_pad=False):
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'r', encoding="utf-8") as fd, \
            open(output_path, 'w+', encoding="utf-8") as out:
        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()

            for w in words:
                if w == '_UNK':
                    break
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ['_PAD', '_UNK'] + \
                sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ['_UNK'] + sorted(vocab, key=vocab.get, reverse=True)

        for v in vocab:
            out.write(v+'\n')


def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path, encoding="utf-8") as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x, y) for (y, x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}


def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if str.isdigit(w) == True:
            w = '0'
        ids.append(vocab.get(w, vocab['_UNK']))

    return ids


def padSentence(s, max_length, vocab):
    return s + [vocab['vocab']['_PAD']]*(max_length - len(s))

# compute f1 score is modified from conlleval.pl


def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                   __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                   (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
               __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
               (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100*correctChunkCnt/foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100*correctChunkCnt/foundCorrectCnt
    else:
        recall = 0

    if (precision+recall) > 0:
        f1 = (2*precision*recall)/(precision+recall)
    else:
        f1 = 0

    return f1, precision, recall


class DataProcessor(object):
    def __init__(self, in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab):
        self.__fd_in = open(in_path, 'r', encoding="utf-8")
        self.__fd_slot = open(slot_path, 'r', encoding="utf-8")
        self.__fd_intent = open(intent_path, 'r', encoding="utf-8")
        self.__in_vocab = in_vocab
        self.__slot_vocab = slot_vocab
        self.__intent_vocab = intent_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_slot.close()
        self.__fd_intent.close()

    def get_batch(self, batch_size):
        in_data = []
        slot_data = []
        slot_weight = []
        length = []
        intents = []

        batch_in = []
        batch_slot = []
        max_len = 0

        # used to record word(not id)
        in_seq = []
        slot_seq = []
        intent_seq = []
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            slot = self.__fd_slot.readline()
            intent = self.__fd_intent.readline()
            inp = inp.rstrip()
            slot = slot.rstrip()
            intent = intent.rstrip()

            in_seq.append(inp)
            slot_seq.append(slot)
            intent_seq.append(intent)

            iii = inp
            sss = slot
            inp = sentenceToIds(inp, self.__in_vocab)
            slot = sentenceToIds(slot, self.__slot_vocab)
            intent = sentenceToIds(intent, self.__intent_vocab)
            batch_in.append(np.array(inp))
            batch_slot.append(np.array(slot))
            length.append(len(inp))
            intents.append(intent[0])
            if len(inp) != len(slot):
                print(iii, sss)
                print(inp, slot)
                exit(0)
            if len(inp) > max_len:
                max_len = len(inp)

        length = np.array(length)
        intents = np.array(intents)
        # print(max_len)
        # print('A'*20)
        for i, s in zip(batch_in, batch_slot):
            in_data.append(padSentence(list(i), max_len, self.__in_vocab))
            slot_data.append(padSentence(list(s), max_len, self.__slot_vocab))
            # print(s)
        in_data = np.array(in_data)
        slot_data = np.array(slot_data)
        # print(in_data)
        # print(slot_data)
        # print(type(slot_data))
        for s in slot_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            slot_weight.append(weight)
        slot_weight = np.array(slot_weight)
        return in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq


def calculate_metrics(pred_intents, correct_intents, slot_outputs, correct_slots):
    # calculate accuracy
    pred_intents = np.array(pred_intents)
    correct_intents = np.array(correct_intents)
    accuracy = (pred_intents == correct_intents)
    semantic_error = accuracy
    accuracy = accuracy.astype(float)
    accuracy = np.mean(accuracy)*100.0

    # Calculate Semantic Error
    index = 0
    for t, p in zip(correct_slots, slot_outputs):
        if len(t) != len(p):
            raise ValueError('Error!!')

        for j in range(len(t)):
            if p[j] != t[j]:
                semantic_error[index] = False
                break
        index += 1
    semantic_error = semantic_error.astype(float)
    semantic_error = np.mean(semantic_error)*100.0

    # Calculate F1, precision and recall
    f1, precision, recall = computeF1Score(correct_slots, slot_outputs)

    return f1, precision, recall, accuracy, semantic_error


def create_f1_lists(slot_outputs, intent_output, intent, slots, input_data, seq_length, slot_vocab, in_vocab):
    # values for f1 score
    slot_out = slot_outputs.cpu().detach().numpy()
    intent_out = intent_output.cpu().detach().numpy()
    pred_intents = [np.argmax(i) for i in intent_out]
    correct_intents = intent
    pred_slots = slot_out.reshape(
        (*slots.shape[0:2], -1)
    )
    slot_outputs_pred = []
    correct_slots = []
    input_words = []
    for p, t, i, l in zip(pred_slots, slots, input_data, seq_length):
        p = np.argmax(p, 1)
        tmp_pred = []
        tmp_correct = []
        tmp_input = []
        for j in range(l):
            tmp_pred.append(slot_vocab['rev'][p[j]])
            tmp_correct.append(slot_vocab['rev'][t[j]])
            tmp_input.append(in_vocab['rev'][i[j]])

        slot_outputs_pred.append(tmp_pred)
        correct_slots.append(tmp_correct)
        input_words.append(tmp_input)

    return pred_intents, correct_intents, slot_outputs_pred, correct_slots, input_words


def validate_model(
    model, batch_size, in_path, slot_path, intent_path,
    in_vocab, slot_vocab, intent_vocab,
    slot_loss_fn, intent_loss_fn
):
    data_processor_valid = DataProcessor(
        in_path, slot_path, intent_path,
        in_vocab, slot_vocab, intent_vocab
    )

    epoch_loss = 0.0
    epoch_slot_loss = 0.0
    epoch_intent_loss = 0.0
    steps_in_epoch = 0

    pred_intents = []
    correct_intents = []
    slot_outputs = []
    correct_slots = []
    input_words = []

    # used to gate
    gate_seq = []
    while True:
        in_data, slot_data, slot_weights, length, intents, _, _, _ \
            = data_processor_valid.get_batch(batch_size)

        in_data, slot_data, slot_weights, length, intents \
            = conv_to_tensor(in_data, slot_data, slot_weights, length, intents)

        slot_out, intent_out = model.forward(input_data=in_data)

        slot_loss, intent_loss, total_loss = calculate_loss(
            slot_data, slot_out, slot_weights, slot_loss_fn,
            intent_out, intents, intent_loss_fn, batch_size
        )

        steps_in_epoch += 1
        epoch_loss += total_loss
        epoch_slot_loss += slot_loss
        epoch_intent_loss += intent_loss

        p_i, c_i, s_o, c_o, i_w = create_f1_lists(
            slot_out, intent_out, intents, slot_data, in_data, length, slot_vocab, in_vocab
        )

        pred_intents.extend(p_i)
        correct_intents.extend(c_i)
        slot_outputs.extend(s_o)
        correct_slots.extend(c_o)
        input_words.extend(i_w)

        if data_processor_valid.end == 1:
            break

    data_processor_valid.close()

    f1, precision, recall, accuracy, semantic_error = calculate_metrics(
        pred_intents, correct_intents, slot_outputs, correct_slots
    )

    total_avg_loss = epoch_loss/steps_in_epoch
    slot_avg_loss = epoch_slot_loss/steps_in_epoch
    intent_avg_loss = epoch_intent_loss/steps_in_epoch

    return f1, accuracy, semantic_error, total_avg_loss, slot_avg_loss, intent_avg_loss


def conv_to_tensor(*args: np.ndarray) -> tuple[torch.Tensor, ...]:
    return (torch.tensor(i) for i in args)


def calculate_loss(
    slots, slot_outputs, slot_weights, slot_loss_fn,
    intent_output, intent, intent_loss_fn, batch_size
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slots_shape = slots.size()
    slots_reshape = slots.reshape([-1])
    crossent = slot_loss_fn(slot_outputs, slots_reshape.long())
    crossent = crossent.reshape(slots_shape)
    slot_loss = torch.sum(crossent * slot_weights, 1)
    total_size = torch.sum(slot_weights, 1)
    total_size += 1e-12
    slot_loss = torch.sum(slot_loss / total_size)

    crossent = intent_loss_fn(intent_output, intent.long())
    intent_loss = torch.sum(crossent) / batch_size

    return slot_loss, intent_loss, slot_loss + intent_loss


def log_in_tensorboard(
    tb_log_writer, epoch, type_, total_loss, intent_loss, slot_loss,
    f1_score, accuracy, semantic_error
):

    logging.info('Epoch: ' + str(epoch) + ' ' + type_)
    logging.info('Total Loss: ' + str(total_loss))
    logging.info('Intent Loss: ' + str(intent_loss))
    logging.info('Slot Loss: ' + str(slot_loss))
    logging.info('F1 Score: ' + str(f1_score))
    logging.info('Accuracy: ' + str(accuracy))
    logging.info('Semantic Err: ' + str(semantic_error))

    tb_log_writer.add_scalar(f"{type_}/loss/total", total_loss, epoch)
    tb_log_writer.add_scalar(f"{type_}/loss/intent", intent_loss, epoch)
    tb_log_writer.add_scalar(f"{type_}/loss/slot", slot_loss, epoch)
    tb_log_writer.add_scalar(f"{type_}/f1", f1_score, epoch)
    tb_log_writer.add_scalar(f"{type_}/acc", accuracy, epoch)
    tb_log_writer.add_scalar(f"{type_}/semantic_err", semantic_error, epoch)
