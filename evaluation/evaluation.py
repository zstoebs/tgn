import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import pickle
import os


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()
  results_files = os.listdir('results/edge')
  eval_files = [f for f in results_files if 'edge_eval' in f]
  run_idx = len(eval_files)
  context_dict = {'pos_prob':[],
                    'neg_prob':[],
                    'pred_score':[],
                    'true_label':[],
                    'sources_batch': [],
                    'destinations_batch': [],
                    'negative_samples': [],
                    'timestamps_batch': [],
                    'edge_idxs_batch': [],
                    'n_neighbors': [],
                  }

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))
      context_dict['pos_prob'] += [pos_prob.cpu().numpy()]
      context_dict['neg_prob'] += [neg_prob.cpu().numpy()]
      context_dict['pred_score'] += [pred_score]
      context_dict['true_label'] += [true_label]
      context_dict['sources_batch'] += [sources_batch]
      context_dict['destinations_batch'] += [destinations_batch]
      context_dict['negative_samples'] += [negative_samples]
      context_dict['timestamps_batch'] += [timestamps_batch]
      context_dict['edge_idxs_batch'] += [edge_idxs_batch]
      context_dict['n_neighbors'] += [n_neighbors]
    pickle.dump(context_dict,open('results/edge/edge_eval_%d.pkl'%(run_idx),'wb'))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)
  results_files = os.listdir('results/node')
  eval_files = [f for f in results_files if 'node_eval' in f]
  run_idx = len(eval_files)
  context_dict = {'source_embedding':[],
                    'destination_embedding':[],
                    'pred_prob_batch':[],
                    'pred_prob':[],
                    'sources_batch': [],
                    'destinations_batch': [],
                    'timestamps_batch': [],
                    'edge_idxs_batch': [],
                    'n_neighbors': [],
                  }
                  
  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
      
      context_dict['source_embedding'] += [source_embedding.cpu().numpy()]
      context_dict['destination_embedding'] += [destination_embedding.cpu().numpy()]
      context_dict['pred_prob_batch'] += [pred_prob_batch.cpu().numpy()]
      context_dict['pred_prob'] += [pred_prob]
      context_dict['sources_batch'] += [sources_batch]
      context_dict['destinations_batch'] += [destinations_batch]
      context_dict['timestamps_batch'] += [timestamps_batch]
      context_dict['edge_idxs_batch'] += [edge_idxs_batch]
      context_dict['n_neighbors'] += [n_neighbors]
    pickle.dump(context_dict,open('results/node/node_eval_%d.pkl'%(run_idx),'wb'))

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
