
import gc
import os
import pickle
import shutil

import torch
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score

from .config import Config
from .metrics import sequence_level_accuracy, sequence_level_accuracy_only_sps
from .model import SPCNNClassifier
from .utils import get_validation_labels, prepare_fold_data, ensure_dirs


def train(lr=None, dropout=None, weight_decay=None):

    if lr is not None:
        Config.LR = lr
    if dropout is not None:
        Config.DROPOUT = dropout
    if weight_decay is not None:
        Config.WEIGHT_DECAY = weight_decay

    ensure_dirs()

    print(f"Using device: {Config.DEVICE}")
    print(f"Model save path: {Config.MODEL_SAVE_PATH}")

    # Store results for all folds
    fold_results = {
        'train_losses': [],
        'val_losses': [],
        'best_val_losses': [],
        'fold_numbers': [],
        'val_token_acc': [],
        'val_seq_acc': [],
        'val_seq_accs_only_sps': [],
        'val_mcc': [],
        'val_precision': [],
        'val_recall': [],
        'best_metrics': []
    }

    # outer training loop (fold)
    for fold in range(1, Config.NUM_FOLDS + 1):

        print(f"\n{'='*60}")
        print(f"Starting Fold {fold}/{Config.NUM_FOLDS}")
        print(f"{'='*60}")

        # Prepare data for this fold
        train_loader, val_loader = prepare_fold_data(fold)

        # Get validation label sequences for sequence-level accuracy
        val_label_seqs = get_validation_labels(fold)

        # initialise the classifier (new for each fold)
        model = SPCNNClassifier(
            embedding_dim=Config.EMBEDDING_DIM,
            num_labels=Config.NUM_CLASSES,
            dropout=Config.DROPOUT,
            lstm_hidden=Config.LSTM_HIDDEN,
            lstm_layers=Config.LSTM_LAYERS,
            conv_filters=Config.CONV_FILTERS,
        ).to(Config.DEVICE)

        # enables mixed precision training (fp16) should be bit faster
        scaler = GradScaler()

        # save metrics and stuff
        train_losses = []
        val_losses = []
        val_token_accs = []
        val_seq_accs = []
        val_seq_accs_only_sps = []
        val_mccs = []
        val_precisions = []
        val_recalls = []

        best_val_loss = float('inf')
        best_mcc = -1.0
        best_metrics = {}
        patience_counter = 0


        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LR,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # inner training loop for this fold (epochs for this fold)
        for epoch in range(Config.EPOCHS):

            # Training phase
            model.train()
            pbar = tqdm(train_loader, desc=f"Fold {fold} - Epoch {epoch+1}/{Config.EPOCHS} [Train]", unit="batch")
            total_train_loss = 0

            # iterate through batches
            for batch in pbar:
                try:
                    # save these to gpu
                    embeddings = batch['embeddings'].to(Config.DEVICE)
                    attention_mask = batch['attention_mask'].to(Config.DEVICE)
                    token_labels = batch['labels'].to(Config.DEVICE)

                    optimizer.zero_grad() # clear the gradients from previous batch, will accumulate otherwise
                    loss = model(embeddings, attention_mask, token_labels) # call forward func, crf and ce give loss directly during training

                    scaler.scale(loss).backward() # backpropagation, scales gradients for fp16, i.e. prevents underflow to 0 using multiplier
                    scaler.unscale_(optimizer) # unscales gradients again to not use the inflated values
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prevent gradient exploding -> more stable training
                    scaler.step(optimizer) # weights are updated using computed gradients (this is skipped if gradients are deemed invalid)
                    scaler.update() # adjust the scale factor for next iteration

                    total_train_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

                except RuntimeError as e:
                    print("Error during training:", e)
                    gc.collect()
                    continue

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase with metrics
            model.eval()
            total_val_loss = 0
            val_batches = 0
            all_val_preds = []
            all_val_labels = []

            with torch.no_grad(): # no gradient calculations needed for inference
                for batch in tqdm(val_loader, desc=f"Fold {fold} - Epoch {epoch+1}/{Config.EPOCHS} [Val]", unit="batch"):
                    embeddings = batch['embeddings'].to(Config.DEVICE)
                    attention_mask = batch['attention_mask'].to(Config.DEVICE)
                    token_labels = batch['labels'].to(Config.DEVICE)

                    loss = model(embeddings, attention_mask, token_labels)
                    total_val_loss += loss.item()
                    val_batches += 1

                    # Get predictions for metrics
                    predictions = model(embeddings, attention_mask) # dont give labels -> crf directly gives predictions

                    # Collect valid tokens (skip -100 labels)
                    for pred_seq, label_seq, mask in zip(predictions, token_labels, attention_mask):
                        for pred, true, is_valid in zip(pred_seq, label_seq, mask):
                            if true.item() != -100 and is_valid.item() == 1:
                                all_val_preds.append(pred)
                                all_val_labels.append(true.item())

            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else 0
            val_losses.append(avg_val_loss)

            # Calculate validation metrics
            if len(all_val_preds) > 0:
                token_acc = accuracy_score(all_val_labels, all_val_preds)
                seq_acc = sequence_level_accuracy(all_val_preds, all_val_labels, val_label_seqs)
                seq_acc_only_sps = sequence_level_accuracy_only_sps(all_val_preds, all_val_labels, val_label_seqs)
                mcc = matthews_corrcoef(all_val_labels, all_val_preds)
                precision = precision_score(all_val_labels, all_val_preds, average='weighted', zero_division="warn")
                recall = recall_score(all_val_labels, all_val_preds, average='weighted', zero_division="warn")
            else:
                token_acc = seq_acc = seq_acc_only_sps = mcc = precision = recall = 0.0

            val_token_accs.append(token_acc)
            val_seq_accs.append(seq_acc)
            val_seq_accs_only_sps.append(seq_acc_only_sps)
            val_mccs.append(mcc)
            val_precisions.append(precision)
            val_recalls.append(recall)

            print(f"Fold {fold} - Epoch {epoch+1}/{Config.EPOCHS}")
            print(f"  Loss: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
            print(f"  Metrics: TokenAcc={token_acc:.4f}, SeqAcc={seq_acc:.4f}, SeqAcc(OnlySPs)={seq_acc_only_sps:.4f}, MCC={mcc:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")

            # Save best model for fold based on MCC TODO maybe alter that a bit
            if mcc > best_mcc:
                patience_counter = 0
                best_mcc = mcc
                best_val_loss = avg_val_loss
                best_metrics = {
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                    'token_acc': token_acc,
                    'seq_acc': seq_acc,
                    'seq_acc_only_sps': seq_acc_only_sps,
                    'mcc': mcc,
                    'precision': precision,
                    'recall': recall
                }
                model_path_temp = Config.MODEL_SAVE_PATH_TEMP.format(fold)
                torch.save(model.state_dict(), model_path_temp)
                print(f"-> Best model for fold {fold} saved to {model_path_temp} (MCC={mcc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} for fold {fold}")
                    break

        # Store results for fold
        fold_results['train_losses'].append(train_losses)
        fold_results['val_losses'].append(val_losses)
        fold_results['best_val_losses'].append(best_val_loss)
        fold_results['fold_numbers'].append(fold)
        fold_results['val_token_acc'].append(val_token_accs)
        fold_results['val_seq_acc'].append(val_seq_accs)
        fold_results['val_seq_accs_only_sps'].append(val_seq_accs_only_sps)
        fold_results['val_mcc'].append(val_mccs)
        fold_results['val_precision'].append(val_precisions)
        fold_results['val_recall'].append(val_recalls)
        fold_results['best_metrics'].append(best_metrics)

        print(f"\nFold {fold} Best Metrics (Epoch {best_metrics['epoch']}):")
        print(f"  Val Loss: {best_metrics['val_loss']:.4f}")
        print(f"  Token Acc: {best_metrics['token_acc']:.4f}")
        print(f"  Seq Acc: {best_metrics['seq_acc']:.4f}")
        print(f"  Seq Acc (only SPs): {best_metrics['seq_acc_only_sps']:.4f}")
        print(f"  MCC: {best_metrics['mcc']:.4f}")
        print(f"  Precision: {best_metrics['precision']:.4f}")
        print(f"  Recall: {best_metrics['recall']:.4f}")

    print("\n" + "="*60)
    print("K-Fold Cross Validation Complete!")
    print("="*60)

    print("\nSummary of all folds:")
    for i, (best_loss, best_m) in enumerate(zip(fold_results['best_val_losses'], fold_results['best_metrics']), 1):
        print(f"Fold {i}: Loss={best_loss:.4f}, TokenAcc={best_m['token_acc']:.4f}, SeqAcc={best_m['seq_acc']:.4f}, SeqAcc(OnlySPs)={best_m['seq_acc_only_sps']:.4f}, MCC={best_m['mcc']:.4f}")

    avg_best_val_loss = sum(fold_results['best_val_losses']) / Config.NUM_FOLDS
    avg_token_acc = sum(m['token_acc'] for m in fold_results['best_metrics']) / Config.NUM_FOLDS
    avg_seq_acc = sum(m['seq_acc'] for m in fold_results['best_metrics']) / Config.NUM_FOLDS
    avg_seq_acc_only_sps = sum(m['seq_acc_only_sps'] for m in fold_results['best_metrics']) / Config.NUM_FOLDS
    avg_mcc = sum(m['mcc'] for m in fold_results['best_metrics']) / Config.NUM_FOLDS
    avg_precision = sum(m['precision'] for m in fold_results['best_metrics']) / Config.NUM_FOLDS
    avg_recall = sum(m['recall'] for m in fold_results['best_metrics']) / Config.NUM_FOLDS

    print(f"\nAverage Best Metrics across all folds:")
    print(f"  Val Loss: {avg_best_val_loss:.4f}")
    print(f"  Token Acc: {avg_token_acc:.4f}")
    print(f"  Seq Acc: {avg_seq_acc:.4f}")
    print(f"  Seq Acc (only SPs): {avg_seq_acc_only_sps:.4f}")
    print(f"  MCC: {avg_mcc:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall: {avg_recall:.4f}")

    # Find best fold based on MCC
    best_fold_mccs = [m['mcc'] for m in fold_results['best_metrics']]
    best_fold_idx = best_fold_mccs.index(max(best_fold_mccs))
    best_fold_num = fold_results['fold_numbers'][best_fold_idx]
    best_fold_mcc = best_fold_mccs[best_fold_idx]
    best_fold_loss = fold_results['best_val_losses'][best_fold_idx]

    print(f"\n{'='*60}")
    print(f"Best performing fold: Fold {best_fold_num} (selected by MCC)")
    print(f"Best MCC: {best_fold_mcc:.4f}")
    print(f"Corresponding validation loss: {best_fold_loss:.4f}")
    print(f"{'='*60}")

    # Copy best model to final path
    best_model_path = Config.MODEL_SAVE_PATH_TEMP.format(best_fold_num)
    shutil.copy(best_model_path, Config.MODEL_SAVE_PATH)
    print(f"\nBest model (Fold {best_fold_num}) saved to: {Config.MODEL_SAVE_PATH}")

    # List temporary fold models
    print("\nTemporary fold models:")
    for i in range(1, Config.NUM_FOLDS + 1):
        temp_path = Config.MODEL_SAVE_PATH_TEMP.format(i)
        if os.path.exists(temp_path):
            print(f"  - {temp_path}")

    losses_data = {
        'fold_numbers': fold_results['fold_numbers'],
        'train_losses': fold_results['train_losses'],
        'val_losses': fold_results['val_losses'],
        'best_val_losses': fold_results['best_val_losses'],
        'val_token_acc': fold_results['val_token_acc'],
        'val_seq_acc': fold_results['val_seq_acc'],
        'val_seq_accs_only_sps': fold_results['val_seq_accs_only_sps'],
        'val_mcc': fold_results['val_mcc'],
        'val_precision': fold_results['val_precision'],
        'val_recall': fold_results['val_recall'],
        'best_metrics': fold_results['best_metrics'],
        'best_fold_num': best_fold_num,
        'best_fold_mcc': best_fold_mcc,
        'best_fold_loss': best_fold_loss,
        'avg_best_val_loss': avg_best_val_loss,
        'avg_token_acc': avg_token_acc,
        'avg_seq_acc': avg_seq_acc,
        'avg_seq_acc_only_sps': avg_seq_acc_only_sps,
        'avg_mcc': avg_mcc,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'epochs': Config.EPOCHS,
        'num_folds': Config.NUM_FOLDS
    }

    # open pkl and save all metrics
    with open(Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH, 'wb') as f:
        pickle.dump(losses_data, f)

    print(f"\nTraining results saved to: {Config.TRAIN_VAL_LOSSES_PKL_SAVE_PATH}")

    return avg_mcc


if __name__ == "__main__":
    train()
