import torch
import torch.nn as nn
import numpy as np

class BidirectionalRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        slot_size: int,
        intent_size: int,
        embedding_dim: int = None,
        layer_size: int = 128,
        isTraining: bool = True,
        remove_slot_attn: bool = False,
        add_final_state_to_intent: bool = True
    ) -> None:

        super(BidirectionalRNN, self).__init__()

        self.input_size = input_size
        self.sequence_length = sequence_length
        self.slot_size = slot_size
        self.intent_size = intent_size

        self.embedding_dim = embedding_dim \
            if embedding_dim is not None else layer_size
        self.layer_size = layer_size
        self.isTraining = isTraining
        self.remove_slot_attn = remove_slot_attn
        self.add_final_state_to_intent = add_final_state_to_intent

        self.embedding = nn.Embedding(
            num_embeddings= input_size, 
            embedding_dim= self.embedding_dim
        )

        if isTraining:
            self.bi_lstm = nn.LSTM(
                self.embedding_dim, layer_size, dropout= 0.5,
                bidirectional= True, batch_first= True
            )
        else:
            self.bi_lstm = nn.LSTM(
                embedding_dim, layer_size, 
                bidirectional= True, batch_first= True
            )
        
        self.slot_attn_conv_layer = nn.Conv2d(
            in_channels= layer_size * 2,
            out_channels= layer_size * 2,
            kernel_size= 1,
            stride= 1
        )

        self.slot_attn_lin_layer = nn.Linear(
            in_features= layer_size * 2,
            out_features= layer_size * 2,
            bias= True
        )
        
        self.intent_attn_conv_layer = nn.Conv2d(
            in_channels= layer_size * 2,
            out_channels= layer_size * 2,
            kernel_size= 1,
            stride= 1
        )

        self.intent_attn_lin_layer = nn.Linear(
            in_features= layer_size * 4,
            out_features= layer_size * 2,
            bias= True
        )

        self.slot_gate_lin_layer = nn.Linear(
            in_features= layer_size * 6,
            out_features= layer_size * 2,
            bias= True
        )

        self.intent_proj_lin_layer = nn.Linear(
            in_features= layer_size * 6,
            out_features= intent_size,
            bias= True
        )

        self.slot_proj_lin_layer = nn.Linear(
            in_features= layer_size * 4,
            out_features= slot_size,
            bias= True
        )
    
    def _slot_attn_forward(self, state_outputs, batch_size, num_features):
        state_shape = state_outputs.size()

        slot_inputs = state_outputs
        attn_size = state_shape[2]
        if self.remove_slot_attn == False:
            origin_shape = state_outputs.size()
            hidden = torch.unsqueeze(state_outputs, 1)
            hidden_conv = torch.unsqueeze(state_outputs, 2)
            hidden_conv = hidden_conv.reshape(
                (batch_size, self.layer_size * 2, num_features, 1)
            )
            hidden_features = self.slot_attn_conv_layer(hidden_conv)
            hidden_features = hidden_features.reshape(origin_shape)
            hidden_features = torch.unsqueeze(hidden_features, 1)

            slot_inputs_shape = slot_inputs.size()
            slot_inputs = slot_inputs.reshape([-1, attn_size])
            y = self.slot_attn_lin_layer(slot_inputs)
            y = y.reshape(slot_inputs_shape)
            y = torch.unsqueeze(y, 2)
            
            v = torch.zeros([attn_size], requires_grad= True)
            s = torch.sum(
                v * torch.tanh(hidden_features + y), [3]
            )
            a = torch.softmax(s, -1)
            a = torch.unsqueeze(a, -1)
            slot_d = torch.sum(a * hidden, [2])
            
            return slot_inputs, slot_d

        else:
            slot_inputs = slot_inputs.reshape((-1, attn_size))
            return slot_inputs, None

    def _intent_attn_forward(self, state_outputs, intent_input, batch_size, num_features):
        attn_size = state_outputs.size(2)
        hidden = torch.unsqueeze(state_outputs, 2)
        origin_shape = hidden.size()
        hidden_features = self.intent_attn_conv_layer(
            hidden.reshape(
                (batch_size, self.layer_size * 2, num_features, 1)
            )
        )
        hidden_features = hidden_features.reshape(origin_shape)

        y = self.intent_attn_lin_layer(intent_input)
        y = y.reshape((-1, 1, 1, attn_size))

        v = torch.zeros([attn_size], requires_grad= True)
        s = torch.sum(
            v * torch.tanh(hidden_features + y), [2, 3]
        )

        a = torch.softmax(s, -1)
        a = torch.unsqueeze(a, -1)
        a = torch.unsqueeze(a, -1)
        d = torch.sum(a * hidden, [1, 2])

        if self.add_final_state_to_intent == True:
            intent_output = torch.cat([d, intent_input], 1)
        else:
            intent_output = d

        return intent_output

    def _slot_gated_forward(self, state_outputs, intent_output, slot_inputs, slot_d= None):
        attn_size = state_outputs.size(2)
        intent_gate = self.slot_gate_lin_layer(intent_output)
        intent_gate = intent_gate.reshape([-1, 1, intent_gate.size(1)])
        v = torch.zeros([attn_size], requires_grad= True)
        if self.remove_slot_attn == False:
            slot_gate = v * torch.tanh(slot_d + intent_gate)
        else:
            slot_gate = v * torch.tanh(state_outputs + intent_gate)
        slot_gate = torch.sum(slot_gate, [2])
        slot_gate = torch.unsqueeze(slot_gate, -1)
        if self.remove_slot_attn == False:
            slot_gate = slot_d * slot_gate
        else:
            slot_gate = state_outputs * slot_gate
        slot_gate = slot_gate.reshape([-1, attn_size])
        slot_output = torch.cat([slot_gate, slot_inputs], 1)

        return slot_output

    def forward(self, input_data: torch.Tensor):

        batch_size, num_features = input_data.shape

        input_data = self.embedding(input_data)

        state_outputs, final_state = self.bi_lstm.forward(
            input_data
        )

        final_state = torch.cat(
            [
                final_state[0][0], final_state[0][1], 
                final_state[1][0], final_state[1][1]
            ], 1
        )

        # attention
        # slot attn
        slot_inputs, slot_d = self._slot_attn_forward(
            state_outputs,
            batch_size, num_features
        )

        # intent attn
        intent_output = self._intent_attn_forward(
            state_outputs, final_state,
            batch_size, num_features
        )

        slot_output = self._slot_gated_forward(
            state_outputs, intent_output, slot_inputs, slot_d
        )

        intent = self.intent_proj_lin_layer(intent_output)
        slot = self.slot_proj_lin_layer(slot_output)

        outputs = (slot, intent)
        return outputs



# if __name__ == "__main__":
#     embedding_dim = 64
#     layer_size = 64
#     batch_size = 16
#     num_features = 33
    
#     model = BidirectionalRNN(
#         input_size= input_size,
#         sequence_length= num_features,
#         slot_size= 122,
#         intent_size= 23,
#         num_features= num_features,
#         batch_size= batch_size,
#         embedding_dim= embedding_dim,
#         layer_size= layer_size
#     )
    
#     input_data = np.zeros(
#         (batch_size, num_features, embedding_dim), 
#         dtype= np.float32
#     )

#     out = model.forward(input_data)