import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm import tqdm 
from tqdm import trange
from sklearn.metrics import accuracy_score
from ConvAutoEncoder import ConvAutoencoder
from torch.utils.tensorboard import SummaryWriter


def train(model, train_loader, dev_loader, args, device):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses = []
    loss_history = []
    no_improvement = 0
    for epoch in trange(args.epochs, desc="Epoch"):
        model.train()
        train_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_loader, desc="Training iteration")):

            # TODO: NEED TO CHANGE THIS ACCORDING TO DATA LOADER
            batch = tuple(t.to(device) for t in batch)
            speech, transcription, sex_label = batch

            optimizer.zero_grad()

            # forward
            reconstructed_speech, sex_logits = model(speech)
            recon_loss = args.reconstruction_loss(reconstructed_speech, speech)
            sex_loss = args.sex_classification_loss(sex_logits, sex_label)

            # backward
            recon_loss.backward()
            sex_loss.backward()
            train_loss += args.r_weight * recon_loss.item() + args.s_weight * sex_loss.item()
            optimizer.step()

        # print out losses
        print("Loss history:", train_losses)
        print("Train loss:", train_loss/nb_tr_steps)

        dev_loss, _, _ = evaluate(model, dev_loader, device="cuda")
        print("Dev loss:", dev_loss)

        if len(loss_history) == 0 or dev_loss < min(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), args.output_model_file)
        else:
            no_improvement += 1
        
        if no_improvement >= args.patience:
            print("No improvement on development set. Finish training.")
            break

        train_losses.append(train_loss/ len(train_loader))
        loss_history.append(dev_loss)

        args.writer.add_scalar("Loss/train", train_loss/len(train_loader), epoch)
        args.writer.add_scalar("Loss/dev", dev_loss, epoch)
    
    return train_losses, loss_history

        

def evaluate(model, dev_loader, args, device):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    model.to(device)
    for step, batch in enumerate(tqdm(dev_loader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        speech, transcription, sex_label = batch

        with torch.no_grad():
            reconstructed_speech, sex_logits = model(speech)

        recon_loss = args.reconstruction_loss(reconstructed_speech, speech)
        sex_loss = args.sex_classification_loss(sex_logits, sex_label)

        outputs = np.argmax(sex_logits.to('cpu'), axis=1)
        sex_label = sex_label.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(sex_label)
        
        eval_loss += args.r_weight * recon_loss.item() + args.s_weight * sex_loss.item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    print("Accuracy on testset: "+str(accuracy_score(correct_labels, predicted_labels)))
        
    return eval_loss, correct_labels, predicted_labels



def test(model, test_loader, args, device):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []
    model.to(device)
    for step, batch in enumerate(tqdm(test_loader, desc="Testing iteration")):
        batch = tuple(t.to(device) for t in batch)
        speech, transcription, sex_label = batch

        with torch.no_grad():
            reconstructed_speech, sex_logits = model(speech)

        recon_loss = args.reconstruction_loss(reconstructed_speech, speech)

        outputs = np.argmax(sex_logits.to('cpu'), axis=1)
        sex_label = sex_label.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(sex_label)
        
        eval_loss += recon_loss.item() 
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    accuracy = accuracy_score(correct_labels, predicted_labels)*100
    print("Test Accuracy: "+str(accuracy))
        
    return eval_loss, accuracy



def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--r_weight', type=float, default=0.5, help='reconstruction loss weight')
    parser.add_argument('--s_weight', type=float, default=0.5, help='sex classification loss weight')
    #parser.add_argument('--a_weight', type=float, default=0.5, help='asr loss weight')
    parser.add_argument('--feature_dim', type=int, default=20, help='input feature dim')
    parser.add_argument('--output_model_file', type=str, default="/checkpoints/model.bin", help='output path for model')
    parser.add_argument('--patience', type=int, default=2)
    
    args = parser.parse_args()

    args.reconstruction_loss = nn.L1Loss(reduction='mean')
    args.sex_classification_loss = nn.NLLLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConvAutoencoder(args.feature_dim)

    writer = SummaryWriter()
    args.writer = writer

    ## train the model ##
    train_losses, dev_losses = train(model, train_loader, dev_loader, args, device)

    ## test the model ##
    test(model, test_loader, args, device)



if __name__ == "__main__":
    main()

