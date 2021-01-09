# Matplotlib chooses Xwindows backend by default. 
# Setting matplotlib to not use the Xwindows backend.
#import matplotlib
#matplotlib.use('Agg')  # this code must be add to the start of script (before importing pyplot)

#import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from parser import parameter_parser
from utils import load_graphsage_data
from model import Autoencoder
# from debug import *


#################################
# Set random seed
seed = 1
np.random.seed(seed)


#################################
# args
args = parameter_parser()

if isinstance(args.learning_rate, float):
    args.learning_rate = [args.learning_rate]

if isinstance(args.batch_size, int):
    args.batch_size = [args.batch_size]


#################################
# carregando o dataset
num_data, feats, labels, train_ids, val_ids, test_ids = load_graphsage_data(args.dataset_path, args.dataset_name)
# print(feats.shape, labels.shape, train_ids.shape, val_ids.shape, test_ids.shape)


#################################
# treinamento
all_params = list()
best_params = None
min_val_loss = 99999999
#best_params_path = 'best_params/pubmed_model'

print('Training started...\n')
for lr in args.learning_rate:
    model = Autoencoder(feats.shape[1],
                        hidden_dim=labels.shape[1],
                        learning_rate=lr)
    
    for b_size in args.batch_size:
        # print('Learning rate: {}, batch size: {}'.format(lr, b_size))
        train_loss, val_loss, time_per_epoch = model.train(feats[train_ids],
                                                            labels[train_ids],
                                                            feats[val_ids],
                                                            labels[val_ids],
                                                            batch_size=b_size,
                                                            epochs=args.epochs,
                                                            early_stopping=args.early_stopping,
                                                            verbose=args.verbose)
        # print('Val loss: {}'.format(val_loss[-1]))
        # print('Training time: {} seconds'.format(round(np.sum(time_per_epoch),4)))
        
        # precisão
        preds = model.predict(feats[test_ids])
        y_test = np.argmax(labels[test_ids], axis=1)
        y_pred = np.argmax(preds[0], axis=1)
        micro_f1 = f1_score(y_test, y_pred, average='micro')
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print('lr: {}, batch_size: {}'.format(lr, b_size))
        print("macro_f1: {}, micro_f1:{}\n".format(macro_f1, micro_f1))
        
        # best model
        if val_loss[-1][0] < min_val_loss:
            min_val_loss = val_loss[-1][0]
            best_params = [lr, b_size, train_loss, val_loss, time_per_epoch, micro_f1, macro_f1]
            # model.save_model(best_params_path)
        
        all_params.append([lr, b_size, train_loss, val_loss, time_per_epoch, micro_f1, macro_f1])

print('Training finished successfully!\n')


#################################
# Melhor model
lr, b_size, train_loss, val_loss, time_per_epoch, micro_f1, macro_f1 = best_params
print('Best model')
print('lr:{}, b_size:{}'.format(lr, b_size))
print('val_loss:', val_loss[-1])
print('average time per epoch: {} seconds'.format(round(np.mean(time_per_epoch),4)))
print("macro_f1: {}, micro_f1:{}".format(macro_f1, micro_f1))


#################################
# Armazenando os resultados (desabilitado para processar o mpprof)
#np.save('results/{}/all_params'.format(args.dataset_name), all_params)
#np.save('results/{}/best_params'.format(args.dataset_name), best_params)
#print('Results saved in the "results" directory.\n')



#################################
###       NÃO UTILIZADO       ###
#################################
# Convergencia
#x = list(range(1,len(train_loss)+1))

#_, ax = plt.subplots()  # Create a figure containing a single axes.
#ax.set_title('Convergência', fontsize=16)
#ax.set_xlabel('Época', fontsize=13)
#ax.set_ylabel('Erro', fontsize=13)
#ax.plot(x, train_loss, label='treino')
#ax.plot(x, val_loss, label='validação')
#plt.legend()
#plt.savefig('results/convergencia.png')
#plt.close()


#################################
# Hints
# EU TENHO QUE CONSEGUIR SALVAR O MODELO PARA PODER EXECUTAR ESSA PARTE

# model = tf.train.restore(....)
#preds = model.predict(feats[test_ids])
#equals = np.equal(np.argmax(preds[0], axis=1), np.argmax(labels[test_ids], axis=1))
#hints = np.sum(equals.astype(int))

#print('Hints: {}/{}'.format(hints, labels[test_ids].shape[0]))

#y_test = np.argmax(labels[test_ids], axis=1)
#y_pred = np.argmax(preds[0], axis=1)
#print("macro_f1: {}, micro_f1:{}".format(f1_score(y_test, y_pred, average='micro'),
#                                             f1_score(y_test, y_pred, average='macro')))

#print('Precision of {}%'.format(round(hints*100/labels[test_ids].shape[0], 3))  )