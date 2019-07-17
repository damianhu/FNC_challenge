import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from feature_engineering import refuting_features, polarity_features, hand_features, gen_or_load_feats
from feature_engineering import word_overlap_features
from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from keras.layers import Embedding, Input
from keras.layers.recurrent import LSTM
from keras.layers.merge import concatenate
from keras.layers.core import Dense
from keras.models import Model, load_model
from utils.system import parse_params, check_version
from keras import optimizers

from keras.callbacks import Callback

from utils.score import LABELS, score_submission


def append_to_loss_monitor_file(text, filepath):
    with open(filepath, 'a+') as the_file:
        the_file.write(text+"\n")

class EarlyStoppingOnF1(Callback):
    """
    Prints some metrics after each epoch in order to observe overfitting
                https://github.com/fchollet/keras/issues/5794
                custom metrics: https://github.com/fchollet/keras/issues/2607
    """

    def __init__(self, epochs,
                 X_test_claims,
                 X_test_orig_docs,
                 y_test, loss_filename, epsilon=0.0, min_epoch=15, X_test_nt=None):
        self.epochs = epochs
        self.patience = 2
        self.counter = 0
        self.prev_score = 0
        self.epsilon = epsilon
        self.loss_filename = loss_filename
        self.min_epoch = min_epoch
        self.X_test_nt = X_test_nt
        # self.print_train_f1 = print_train_f1

        # self.X_train_claims = X_train_claims
        # self.X_train_orig_docs = X_train_orig_docs
        # self.X_train_evid = X_train_evid
        # self.y_train = y_train

        self.X_test_claims = X_test_claims
        self.X_test_orig_docs = X_test_orig_docs
        self.y_test = y_test
        Callback.__init__(self)

    def on_epoch_end(self, epoch, logs={}):
        if epoch + 1 < self.epochs:
            from sklearn.metrics import f1_score

            # get prediction and convert into list
            if type(self.X_test_orig_docs).__module__ == np.__name__ and type(self.X_test_nt).__module__ == np.__name__:
                predicted_one_hot = self.model.predict([
                    self.X_test_claims,
                    self.X_test_orig_docs,
                    self.X_test_nt
                ])
            elif type(self.X_test_orig_docs).__module__ == np.__name__:
                predicted_one_hot = self.model.predict([
                    self.X_test_claims,
                    self.X_test_orig_docs,
                ])
            else:
                predicted_one_hot = self.model.predict(self.X_test_claims)
            predict = np.argmax(predicted_one_hot, axis=-1)

            """
            predicted_one_hot_train = self.model.predict([self.X_train_claims, self.X_train_orig_docs, self.X_train_evid])
            predict_train = np.argmax(predicted_one_hot_train, axis=-1)


            # f1 for train data
            f1_macro_train = ""
            if self.print_train_f1 == True:
                f1_0_train = f1_score(self.y_train, predict_train, labels=[0], average=None)
                f1_1_train = f1_score(self.y_train, predict_train, labels=[1], average=None)
                f1_macro_train = (f1_0_train[0] + f1_1_train[0]) / 2
                print(" - train_f1_(macro): " + str(f1_macro_train))"""

            predicted = [LABELS[int(a)] for a in predict]
            actual = [LABELS[int(a)] for a in self.y_test]
            # calc FNC score
            fold_score, _ = score_submission(actual, predicted)
            max_fold_score, _ = score_submission(actual, actual)
            fnc_score = fold_score / max_fold_score
            print(" - fnc_score: " + str(fnc_score))

            # f1 for test data
            f1_0 = f1_score(self.y_test, predict, labels=[0], average=None)
            f1_1 = f1_score(self.y_test, predict, labels=[1], average=None)
            f1_2 = f1_score(self.y_test, predict, labels=[2], average=None)
            f1_3 = f1_score(self.y_test, predict, labels=[3], average=None)
            f1_macro = (f1_0[0] + f1_1[0] + f1_2[0] + f1_3[0]) / 4
            print(" - val_f1_(macro): " + str(f1_macro))
            print("\n")

            header = ""
            values = ""
            for key, value in logs.items():
                header = header + key + ";"
                values = values + str(value) + ";"
            if epoch == 0:
                values = "\n" + header + "val_f1_macro;" + "fnc_score;" + "\n" + values + str(f1_macro) + str(
                    fnc_score) + ";"
            else:
                values += str(f1_macro) + ";" + str(fnc_score) + ";"
            append_to_loss_monitor_file(values, self.loss_filename)

            if epoch >= self.min_epoch - 1:  # 9
                if f1_macro + self.epsilon <= self.prev_score:
                    self.counter += 1
                else:
                    self.counter = 0
                if self.counter >= 2:
                    self.model.stop_training = True
            # print("Counter at " + str(self.counter))
            self.prev_score = f1_macro
            # print("\n")

def calculate_class_weight(y_train, no_classes=2):
    # https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
    from sklearn.utils import class_weight

    class_weight_list = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = {}
    for i in range(no_classes):
        class_weights[i] = class_weight_list[i]
    print(class_weights)
    return class_weights

def convert_data_to_one_hot(y_train):
    # y_test_temp = np.zeros((y_test.size, y_test.max() + 1), dtype=np.int)
    # y_test_temp[np.arange(y_test.size), y_test] = 1

    # Other option:
    #   y_train is a tensor then because of one_hot, but feed_dict only accepts numpy arrays => replace y_train with sess.run(y_train)
    #   http://stackoverflow.com/questions/34410654/tensorflow-valueerror-setting-an-array-element-with-a-sequence
    # return tf.one_hot(y_train, 4), tf.one_hot(y_test, 4)
    y_train_temp = np.zeros((y_train.size, y_train.max() + 1), dtype=np.int)
    y_train_temp[np.arange(y_train.size), y_train] = 1

    return y_train_temp

def split_X(X_train, MAX_SEQ_LENGTH_HEADS):
    # split to get [heads, docs]
    X_train_splits = np.hsplit(X_train, np.array([MAX_SEQ_LENGTH_HEADS]))
    X_train_head = X_train_splits[0]
    X_train_doc = X_train_splits[1]

    print("X_train_head.shape = " + str(np.array(X_train_head).shape))
    print("X_train_doc.shape = " + str(np.array(X_train_doc).shape))

    return X_train_head,X_train_doc

def generate_features(stances,dataset,name):
    h, b, y = [],[],[]

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X_overlap = gen_or_load_feats(word_overlap_features, h, b, "features/overlap."+name+".npy")
    X_refuting = gen_or_load_feats(refuting_features, h, b, "features/refuting."+name+".npy")
    X_polarity = gen_or_load_feats(polarity_features, h, b, "features/polarity."+name+".npy")
    X_hand = gen_or_load_feats(hand_features, h, b, "features/hand."+name+".npy")

    X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
    return X,y

if __name__ == "__main__":

    batch_size = 128
    epochs = 100
    word_index = 100
    EMBEDDING_DIM = 50
    MAX_LENGTH = 16
    LSTM_implementation = 2
    embeddings_index = {}
    from gensim.models import KeyedVectors
    #
    glove_file = 'glove.twitter.27B.50d.txt'
    tmp_file = 'word2vec.txt'

    vecmodel = KeyedVectors.load_word2vec_format(tmp_file)
    word2idx = {'_PAD': 0}

    vocab_list = [(k, vecmodel.wv[k]) for k, v in vecmodel.wv.vocab.items()]
    embeddings_matrix = np.zeros((len(vecmodel.wv.vocab.items()) + 1, vecmodel.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
    print(embeddings_matrix.shape)

    # embedding_layer = Embedding(len(vecmodel.wv.vocab.items()) + 1,
    #                             EMBEDDING_DIM,
    #                             weights=[embeddings_matrix],
    #                             trainable=False,
    #                             input_length=4
    #                             )
    # parse_params()
    # competition_dataset = DataSet("competition_test")
    # X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")
    d = DataSet()
    folds, hold_out = kfold_split(d, n_folds=10)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)

    # Load the competition dataset
    competition_dataset = DataSet("competition_test")
    X_competition, y_competition = generate_features(competition_dataset.stances, competition_dataset, "competition")

    Xs = dict()
    ys = dict()

    # Load/Precompute all features now
    X_holdout, y_holdout = generate_features(hold_out_stances, d, "holdout")
    for fold in fold_stances:
        Xs[fold], ys[fold] = generate_features(fold_stances[fold], d, str(fold))

    best_score = 0
    best_fold = None

    for fold in fold_stances:
        ids = list(range(len(folds)))
        del ids[fold]

        X_train = np.vstack(tuple([Xs[i] for i in ids]))
        y_train = np.hstack(tuple([ys[i] for i in ids]))

        X_test = Xs[fold]
        y_test = ys[fold]
        y_train_one_hot = convert_data_to_one_hot(y_train)
        y_test_one_hot = convert_data_to_one_hot(np.array(y_test))
        X_train_LSTM, X_train_MLP = split_X(X_train, MAX_LENGTH)
        X_test_LSTM, X_test_MLP = split_X(X_test, MAX_LENGTH)
        class_weights = calculate_class_weight(y_train, no_classes=4)
        lstm_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='lstm_input')
        embedding = Embedding(input_dim=len(vecmodel.wv.vocab.items())+1,  # lookup table size
                              output_dim=EMBEDDING_DIM,  # output dim for each number in a sequence
                              weights=[embeddings_matrix],
                              input_length=MAX_LENGTH,  # receive sequences of MAX_SEQ_LENGTH_CLAIMS integers
                              mask_zero=True,
                              trainable=True)(lstm_input)

        data_LSTM = LSTM(
            100, return_sequences=True, stateful=False, dropout=0.2,
            batch_input_shape=(batch_size, MAX_LENGTH, EMBEDDING_DIM),
            input_shape=(MAX_LENGTH, EMBEDDING_DIM), implementation=LSTM_implementation
        )(embedding)
        data_LSTM = LSTM(
            100, return_sequences=False, stateful=False, dropout=0.2,
            batch_input_shape=(batch_size, MAX_LENGTH, EMBEDDING_DIM),
            input_shape=(MAX_LENGTH, EMBEDDING_DIM), implementation=LSTM_implementation
        )(data_LSTM)

        mlp_input = Input(shape=(len(X_train_MLP[0]),), dtype='float32', name='mlp_input')

        merged = concatenate([data_LSTM, mlp_input])

        dense_mid = Dense(600, kernel_regularizer=None, kernel_initializer='glorot_uniform',
                          activity_regularizer= None, activation='relu')(merged)
        dense_mid = Dense(600, kernel_regularizer=None, kernel_initializer='glorot_uniform',
                          activity_regularizer=None, activation='relu')(dense_mid)
        dense_mid = Dense(600, kernel_regularizer=None, kernel_initializer='glorot_uniform',
                          activity_regularizer=None, activation='relu')(dense_mid)
        dense_out = Dense(4, activation='softmax', name='dense_out')(dense_mid)

        model = Model(inputs=[lstm_input, mlp_input], outputs=[dense_out])

        model.compile(optimizer=optimizers.Adam(), loss='kullback_leibler_divergence',  # categorial_crossentropy
                           metrics=['accuracy']
                           )

        print("Starting training ")
        print("xtrain shape:" + str(np.array(X_train).shape))
        print("ytrain shape:" + str(np.array(y_train).shape))
        model.fit([X_train_LSTM, X_train_MLP],
                       y_train_one_hot,
                       validation_data=([X_test_LSTM, X_test_MLP], y_test_one_hot),
                       batch_size=batch_size, epochs=epochs, verbose=1,
                       callbacks=[
                           EarlyStoppingOnF1(epochs,
                                             X_test_LSTM, X_test_MLP, y_test,
                                             'model_loss', epsilon=0.0, min_epoch=10),
                       ]
        )
        print("Training finished \n")
        X_test_LSTM, X_test_MLP = split_X(X_test, MAX_LENGTH)
        predicted_one_hot = model.predict(x=[X_test_LSTM, X_test_MLP])
        predicted = np.argmax(predicted_one_hot, axis=-1)
        y_test = np.argmax(y_test, axis=-1)

        predicted = [LABELS[int(a)] for a in predicted]
        actual = [LABELS[int(a)] for a in y_test]
        fold_score, _ = score_submission(actual, predicted)
        max_fold_score, _ = score_submission(actual, actual)
        score = fold_score / max_fold_score

        print("Score for fold " + str(fold) + " was - " + str(score))
        if score > best_score:
            best_score = score
            best_fold = model
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_holdout)]
    actual = [LABELS[int(a)] for a in y_holdout]
    print("Scores on the dev set")
    report_score(actual, predicted)
    print("")
    print("")

    # Run on competition dataset
    predicted = [LABELS[int(a)] for a in best_fold.predict(X_competition)]
    actual = [LABELS[int(a)] for a in y_competition]

    print("Scores on the test set")
    report_score(actual, predicted)








