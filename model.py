from keras.models import Model
from keras.layers import Input, concatenate, multiply, Activation
from keras.layers import Dense, Embedding, Reshape, GRU, BatchNormalization, Dropout
from keras.optimizers import RMSprop


class ImageCaptionModel:
    """
    Reference : https://github.com/LemonATsu/Keras-Image-Caption/blob/master/model.py
    """

    def __init__(self, vocab_size=2187, embedding_matrix=None, language_dim=100,
                 max_caplen=53, img_dim=2048, clipnorm=1.):

        self.vocab_size = vocab_size
        self.embed_mat = embedding_matrix
        self.lang_dim = language_dim

        self.max_cap_len = max_caplen
        self.img_dim = img_dim

        # hyper-parameter
        self.do_rate = .25
        self.gru_units = 1024
        self.lr = 1e-4
        self.clip_norm = clipnorm

        self.build_model()

    def build_model(self):
        # Input
        lang_input = Input(shape=(1,))
        img_input = Input(shape=(self.img_dim,))
        seq_input = Input(shape=(self.max_cap_len,))
        vocab_input = Input(shape=(self.vocab_size,))

        if self.embed_mat:
            x = Embedding(input_dim=self.vocab_size, output_dim=self.lang_dim,
                          embeddings_initializer='glorot_uniform',
                          input_length=1,
                          weights=[self.embed_mat])(lang_input)
        else:
            x = Embedding(input_dim=self.vocab_size, output_dim=self.lang_dim,
                          embeddings_initializer='glorot_uniform',
                          input_length=1)(lang_input)

        lang_embed = Reshape((self.lang_dim,))(x)
        lang_embed = concatenate([lang_embed, seq_input])
        lang_embed = Dense(self.lang_dim)(lang_embed)
        lang_embed = Dropout(self.do_rate)(lang_embed)

        layers = concatenate([img_input, lang_embed, vocab_input])
        layers = Reshape((1, self.lang_dim + self.img_dim + self.vocab_size))(layers)

        # GRU - 1
        gru_1 = GRU(self.img_dim)(layers)
        gru_1 = Dropout(self.do_rate)(gru_1)
        gru_1 = Dense(self.img_dim)(gru_1)
        gru_1 = BatchNormalization()(gru_1)
        gru_1 = Activation('softmax')(gru_1)

        # soft attention
        attention_1 = multiply([img_input, gru_1])
        attention_1 = concatenate([attention_1, lang_embed, vocab_input])
        attention_1 = Reshape((1, self.lang_dim + self.img_dim + self.vocab_size))(attention_1)

        # GRU - 2
        gru_2 = GRU(self.gru_units)(attention_1)
        gru_2 = Dropout(self.do_rate)(gru_2)
        gru_2 = Dense(self.vocab_size)(gru_2)
        gru_2 = BatchNormalization()(gru_2)

        out = Activation('softmax')(gru_2)

        model = Model(inputs=[img_input, lang_input, seq_input, vocab_input], outputs=out)

        # opt = Adam(lr=self.lr, clipnorm=self.clip_norm)
        opt = RMSprop(lr=self.lr, clipnorm=self.clip_norm)

        model.compile(loss='categorical_crossentropy', optimizer=opt)

        return model
