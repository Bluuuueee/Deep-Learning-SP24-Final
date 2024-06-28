import torch
import torch.nn as nn

from models.ConvLSTMCell import ConvLSTMCell

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.dropout_rate = 0.1
        dropout_rate = self.dropout_rate

        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=nf+in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.dropout2 = nn.Dropout(p=dropout_rate)


        self.encoder_3_convlstm = ConvLSTMCell(input_dim= 2 * nf + in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        
        self.decoder_1_convlstm = ConvLSTMCell(input_dim= 3 * nf + in_chan,  # nf + 1
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.dropout4 = nn.Dropout(p=dropout_rate)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim= 4 * nf + in_chan,
                                               hidden_dim=nf,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.dropout5 = nn.Dropout(p=dropout_rate)
        
        self.decoder_3_convlstm = ConvLSTMCell(input_dim= 5 * nf + in_chan,
                                               hidden_dim= 3 * nf + in_chan,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.dropout6 = nn.Dropout(p=dropout_rate)
        
        self.decoder_CNN = nn.Conv3d(in_channels=3 * nf + in_chan,
                                     out_channels=in_chan,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))


    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6):

        outputs = []

        # encoder
        for t in range(seq_len):
            #print("--------------t: ", t)
            #print("Xt: ", x[:, t, :, :].shape)
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here

            o = torch.cat((x[:, t, :, :], h_t), dim=1)
            o = self.dropout1(o)
            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=o,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here
            #print("ht2: ", h_t2.shape)
            #print("ct2: ", c_t2.shape)
            
            o = torch.cat((x[:, t, :, :], h_t, h_t2), dim=1)
            o = self.dropout2(o)
            h_t3, c_t3 = self.encoder_3_convlstm(input_tensor=o,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            #print("ht3: ", h_t3.shape)
            #print("ct3: ", c_t3.shape)
        # encoder_vector
        encoder_vector = torch.cat((x[:, seq_len-1, :, :], h_t, h_t2, h_t3), dim=1)
        encoder_vector = self.dropout3(encoder_vector)
        #print("enc: ", encoder_vector.shape)

        # decoder
        for t in range(future_step):
            h_t4, c_t4 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            #print("ht4: ", h_t4.shape)
            #print("ct4: ", c_t4.shape)
            
            o = torch.cat((encoder_vector,h_t4), dim=1)
            o = self.dropout4(o)
            h_t5, c_t5 = self.decoder_2_convlstm(input_tensor=o,
                                                 cur_state=[h_t5, c_t5])  # we could concat to provide skip conn here
            #print("ht5: ", h_t5.shape)
            #print("ct5: ", c_t5.shape)

            
            o = torch.cat((encoder_vector,h_t4, h_t5), dim=1)
            o = self.dropout5(o)
            h_t6, c_t6 = self.decoder_3_convlstm(input_tensor=o,
                                                 cur_state=[h_t6, c_t6])  # we could concat to provide skip conn here
            #print("ht6: ", h_t6.shape)
            #print("ct6: ", c_t6.shape)
            encoder_vector = h_t6
            outputs += [self.dropout6(h_t6)]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        # outputs = torch.nn.Sigmoid()(outputs)
        outputs = outputs.permute(0, 2, 3, 4, 1)

        return outputs

    def forward(self, x, future_seq=11, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        # print("TRAINING: ", self.training)
        b, seq_len, _, h, w = x.size()
        # print("FWD: ", b, seq_len, _, h, w)
        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t5, c_t5 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t6, c_t6 = self.decoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6)

        return outputs
