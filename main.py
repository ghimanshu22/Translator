import torch
import torch.optim as optim
from model import Seq2SeqTransformer
from train import train_epoch, evaluate
from translate import translate
from data_processing import PAD_IDX, BOS_IDX, EOS_IDX, SRC_LANGUAGE, TGT_LANGUAGE, vocab_transform, create_mask

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
EMB_SIZE = 512
NHEAD = 8
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
FFN_HID_DIM = 512

# Instantiate the model
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

# Loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001)

# Training loop
NUM_EPOCHS = 10
for epoch in range(1, NUM_EPOCHS+1):
    train_loss = train_epoch(transformer, optimizer)
    val_loss = evaluate(transformer)
    print(f"Epoch {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

# Translate a sentence
sentence = "Eine Gruppe von Menschen steht vor einem Iglu ."
print(translate(transformer, sentence))
