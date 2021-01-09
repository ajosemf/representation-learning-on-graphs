import tensorflow as tf
import numpy as np
import time

tf.logging.set_verbosity(tf.logging.ERROR)

class Autoencoder(object):

    def __init__(self, input_dim, hidden_dim, learning_rate):
        self._learning_rate = learning_rate
        
        # Weights and biases
        w1 = tf.Variable(tf.random_normal([input_dim, hidden_dim]))
        b1 = tf.Variable(tf.random_normal([hidden_dim]))
        w2 = tf.Variable(tf.random_normal([hidden_dim, input_dim]))
        b2 = tf.Variable(tf.random_normal([input_dim]))
        # w3 = tf.Variable(tf.random_normal([hidden_dim, hidden_dim]))
        # b3 = tf.Variable(tf.random_normal([hidden_dim]))
        
        # Placeholder
        self._input_layer = tf.placeholder('float', [None, input_dim])
        self._labels = tf.placeholder('float', [None, hidden_dim])  # (onehot encoding)
        
        # Encoder
        self._hidden_layer = tf.nn.relu(tf.add(tf.matmul(self._input_layer, w1), b1))
        
        # Classifier
        ####### aqui entra o softmax da layer hidden
        # self._preds = tf.nn.softmax(tf.nn.relu(tf.add(tf.matmul(self._hidden_layer, w3), b3)))
        self._preds = tf.nn.softmax(self._hidden_layer)
        
        # Decoder
        self._output_layer = tf.matmul(self._hidden_layer, w2) + b2
        
        # Loss
        self._loss1 = tf.reduce_mean(tf.square(self._output_layer - self._input_layer))
        self._loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._preds, labels=self._labels))
        self._total_loss = self._loss1 + self._loss2
        
        # Optimizer
        self._optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._total_loss)
        
        # Init
        self._init = tf.global_variables_initializer()
        self._session = tf.Session()

        # Set random seed
        seed = 1
        np.random.seed(seed)
        tf.set_random_seed(seed)
        

    # def train(self, input_train, input_test, batch_size, epochs):
    ####### input_test nao esta sendo utilizado
    def train(self, input_train, labels_train, input_val, labels_val, batch_size, epochs, early_stopping, verbose=True):
        """adjust model parameters by minimizing cost function error

        Arguments:
            input_train {np.ndarray} -- [description]
            labels_train {np.ndarray} -- [description]
            input_val {np.ndarray} -- [description]
            labels_val {np.ndarray} -- [description]
            batch_size {int} -- [description]
            epochs {int} -- [description]

        Returns:
            tuple -- tuple of lists (train_loss, val_loss, time_per_epoch) containing the error and training time per epoch
        """
        self._session.run(self._init)
        self._saver = tf.train.Saver()
        
        train_loss = list()
        val_loss = list()
        time_per_epoch = list()
        train_ids = list(range(input_train.shape[0]))

        for epoch in range(epochs):
            start = time.time()
            # t_loss = 0

            # shuffle
            np.random.shuffle(train_ids)

            # optimization
            for i in range(int(input_train.shape[0]/batch_size)):
                batch_ids = train_ids[i * batch_size : (i + 1) * batch_size]
                batch_input = input_train[batch_ids]
                batch_labels = labels_train[batch_ids]
                _, t_loss = self._session.run([self._optimizer, self._total_loss], feed_dict={self._input_layer: batch_input, self._labels: batch_labels})
                #epoch_input = input_train[ i * batch_size : (i + 1) * batch_size ]
                #epoch_labels = labels_train[ i * batch_size : (i + 1) * batch_size ]
                # _, c = self._session.run([self._optimizer, self._total_loss], feed_dict={self._input_layer: epoch_input, self._labels: epoch_labels})
                # t_loss += c

            # validation
            v_loss = self._session.run([self._total_loss], feed_dict={self._input_layer: input_val, self._labels: labels_val})
            # resum
            train_loss.append(t_loss)
            val_loss.append(v_loss)
            time_per_epoch.append(time.time() - start)
            if verbose and ((epoch+1)%10 == 0 or epoch == 0):
                print('Epoch', epoch+1, '/', epochs, 'loss:', t_loss)

            if epoch > early_stopping and val_loss[-1] > np.mean(val_loss[-(early_stopping + 1):-1]):
                if verbose: print('Early stopping at {} epochs\n'.format(epoch+1))
                break

        return train_loss, val_loss, time_per_epoch

    
    def save_model(self, path):
        self._saver.save(self._session, path)
        print('Model saved\n')


    #def load_model(self, path):
        #self._session.run(self._init)
        #self._saver = tf.train.Saver()
        #self._saver.restore(self._session, path)
        #print('Model loaded')


    def predict(self, input):
        return self._session.run([self._preds], feed_dict={self._input_layer: input})
        

    #def getEncodedImage(self, image):
        #encoded_image = self._session.run(self._hidden_layer, feed_dict={self._input_layer:[image]})
        #return encoded_image
    

    #def getDecodedImage(self, image):
        #decoded_image = self._session.run(self._output_layer, feed_dict={self._input_layer:[image]})
        #return decoded_image