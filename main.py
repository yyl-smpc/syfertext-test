# -*- coding: utf-8 -*-
import syft as sy
from syft.generic.string import String
# SyferText imports
import syfertext
from syfertext.pipeline import SimpleTagger

# PyTorch imports
import torch as th
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

# Useful imports
import csv
from sklearn.model_selection import train_test_split
from syft.grid.clients.data_centric_fl_client import DataCentricFLClient
from syft.workers.virtual import VirtualWorker
from syft.grid.private_grid import PrivateGridNetwork

hook = sy.TorchHook(th)
me = hook.local_worker
bob = DataCentricFLClient(hook, 'http://127.0.0.1:5001','Bob', is_client_worker=True) # Bob owns the first dataset
alice = DataCentricFLClient(hook, 'http://alice:5000','Alice', is_client_worker=True) # Alice owns the second dataset
crypto_provider = DataCentricFLClient(hook, 'http://charlie:5002','Charlie', is_client_worker=True) # provides encryption primitive for SMPC
PrivateGridNetwork(bob,alice,crypto_provider)
dataset_path = './data/train.csv'
# store the dataset as a list of dictionaries
# each dictionary has two keys, 'text' and 'label'
# the 'text' element is a PySyft String
# the 'label' element is an integer with 1 for each surgical specialty and a 0 otherwise
dataset_local = []
with open(dataset_path, 'r', encoding='UTF-8') as dataset_file:
    # Create a csv reader object
    reader = csv.DictReader(dataset_file)
    for elem in reader:
        # Create one entry
        # Check if the medical specialty contains 1 (label for surgery)
        # otherwise mark it as 0"
        example = dict(text=String(elem['text'],owner=me),
                       label=1 if elem['label'] == '1' else 0
                       )

        # add to the local dataset
        dataset_local.append(example)

# Create two datasets, one for Bob and another one for Alice
dataset_bob, dataset_alice = train_test_split(dataset_local[:25000], train_size=0.5)

# Now create a validation set for Bob and another one for Alice
train_bob, val_bob = train_test_split(dataset_bob, train_size=0.9)
train_alice, val_alice = train_test_split(dataset_alice, train_size=0.9)


# Make a function that sends the content of each split to a remote worker
def make_remote_dataset(dataset, worker):
    # Got through each example in the dataset
    for example in dataset:
        # Send each transcription text
        example['text'] = example['text'].send(worker)
        # Send each label as a one-hot-encoded vector
        one_hot_label = th.zeros(2).scatter(0, th.Tensor([example['label']]).long(), 1)

        # print for debugging purposes
        # print("mapping",example['label']," to ",one_hot_label)

        # Send the transcription label
        example['label'] = one_hot_label.send(worker)

# Bob's remote dataset
make_remote_dataset(train_bob, bob)
make_remote_dataset(val_bob, bob)

# Alice's remote dataset
make_remote_dataset(train_alice, alice)
make_remote_dataset(val_alice, alice)
# Create a Language object with SyferText
nlp = syfertext.load('en_core_web_lg', owner = me)
use_stop_tagger = True
use_vocab_tagger = True

# Token with these custom tags
# will be excluded from creating
# the Doc vector
excluded_tokens = {}
## Load the list of stop words
with open('./data/clinical-stopwords.txt', 'r') as f:
    stop_words = set(f.read().splitlines())

# Create a simple tagger object to tag stop words
stop_tagger = SimpleTagger(attribute='is_stop',
                           lookups=stop_words,
                           tag=True,
                           default_tag=False,
                           case_sensitive=False
                           )

if use_stop_tagger:
    # Add the stop word to the pipeline
    nlp.add_pipe(name='stop tagger',
                 component=stop_tagger,
                 remote=True
                 )

    # Tokens with 'is_stop' = True are
    # not going to be used when creating the
    # Doc vector
    excluded_tokens['is_stop'] = {True}
## Load list of vocab words
with open('./data/vocab.txt', 'r') as f:
    vocab_words = f.read().splitlines()

# Create a simple tagger object to tag stop words
vocab_tagger = SimpleTagger(attribute = 'is_vocab',
                           lookups = vocab_words,
                           tag = True,
                           default_tag = False,
                           case_sensitive = False
                          )

if use_vocab_tagger:

    # Add the stop word to the pipeline
    nlp.add_pipe(name = 'vocab tagger',
                 component = vocab_tagger,
                 remote = True
                )

    # Tokens with 'is_vocab' = False are
    # not going to be used when creating the
    # Doc vector
    excluded_tokens['is_vocab'] = {False}


class DatasetMTS(Dataset):

    def __init__(self, sets, share_workers, crypto_provider, nlp):
        """Initialize the Dataset object

        Args:
            sets (list): A list containing all training OR
                all validation sets to be used.
            share_workers (list): A list of workers that will
                be used to hold the SMPC shares.
            crypto_provider (worker): A worker that will
                provide SMPC primitives for encryption.
            nlp: This is SyferText's Language object containing
                the preprocessing pipeline.
        """
        self.sets = sets
        self.crypto_provider = crypto_provider
        self.workers = share_workers

        # Create a single dataset unifying all datasets.
        # A property called `self.dataset` is created
        # as a result of this call.
        self._create_dataset()

        # The language model
        self.nlp = nlp

    def __getitem__(self, index):
        """In this function, preprocessing with SyferText
        of one transcription will be triggered. Encryption will also
        be performed and the encrypted vector will be obtained.
        The encrypted label will be computed too.

        Args:
            index (int): This is an integer received by the
                PyTorch DataLoader. It specifies the index of
                the example to be fetched. This actually indexes
                one example in `self.dataset` which pools over
                examples of all the remote datasets.
        """

        # get the example
        example = self.dataset[index]
        # Run the preprocessing pipeline on
        # the transcription text and get a DocPointer object
        doc_ptr = self.nlp(example['text'])

        # Get the encrypted vector embedding for the document
        vector_enc = doc_ptr.get_encrypted_vector(bob,
                                                  alice,
                                                  crypto_provider=self.crypto_provider,
                                                  requires_grad=True,
                                                  excluded_tokens=excluded_tokens
                                                  )

        # Encrypt the target label
        label_enc = example['label'].fix_precision().share(bob,
                                                           alice,
                                                           crypto_provider=self.crypto_provider,
                                                           requires_grad=True
                                                           ).get()

        return vector_enc, label_enc

    def __len__(self):
        """Returns the combined size of all of the
        remote training/validation sets.
        """

        # The size of the combined datasets
        return len(self.dataset)

    def _create_dataset(self):
        """Create a single list unifying examples from all remote datasets
        """

        # Initialize the dataset
        self.dataset = []

        # populate the dataset list
        for dataset in self.sets:
            for example in dataset:
                self.dataset.append(example)

    @staticmethod
    def collate_fn(batch):
        """The collat_fn method to be used by the
        PyTorch data loader.
        """

        # Unzip the batch
        vectors, targets = list(zip(*batch))

        # concatenate the vectors
        vectors = th.stack(vectors)

        # concatenate the labels
        targets = th.stack(targets)

        return vectors, targets
# Instantiate a training Dataset object
trainset = DatasetMTS(sets = [train_bob,train_alice],
                       share_workers = [bob,alice],
                       crypto_provider = crypto_provider,
                       nlp = nlp
                      )

# Instantiate a validation Dataset object
valset = DatasetMTS(sets = [val_bob,val_alice],
                     share_workers = [bob,alice],
                     crypto_provider = crypto_provider,
                     nlp = nlp
                    )
vec_enc, label_enc = trainset.__getitem__(1)
print(f' Training Vector size is {vec_enc.shape[0]}')

EMBED_DIM = vec_enc.shape[0]
BATCH_SIZE = 128 # chunks of data to be passed through the network
LEARNING_RATE = 0.001
EPOCHS = 3 # Complete passes of the entire data
NUN_CLASS = 2 # 2 classes since its a binary classifier

# Instantiate the DataLoader object for the training set
trainloader = DataLoader(trainset, shuffle = True,
                         batch_size = BATCH_SIZE, num_workers = 0,
                         collate_fn = trainset.collate_fn)


# Instantiate the DataLoader object for the validation set
valloader = DataLoader(valset, shuffle = True,
                       batch_size = BATCH_SIZE, num_workers = 0,
                       collate_fn = valset.collate_fn)


class Classifier(th.nn.Module):

    def __init__(self, in_features, out_features):
        super(Classifier, self).__init__()

        self.fc1 = th.nn.Linear(in_features, 64)
        self.fc2 = th.nn.Linear(64, 32)
        self.fc3 = th.nn.Linear(32, out_features)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs.squeeze(1)))
        x = F.relu(self.fc2(x))

        logits = self.fc3(x)

        probs = F.relu(logits)

        return probs, logits
# Create the classifer
model = Classifier(in_features = EMBED_DIM, out_features = NUN_CLASS)

# Apply SMPC encryption
model = model.fix_precision().share(bob, alice,
                                              crypto_provider = crypto_provider,
                                              requires_grad = True
                                              )
print(model)
optimizer = optim.SGD(params = model.parameters(),
                  lr = LEARNING_RATE, momentum=0.3)

optimizer = optimizer.fix_precision()
# Create a summary writer for logging performance with Tensorboard
writer = SummaryWriter()
# save losses for debugging/plotting
train_losses = []
train_acc = []
train_iter = []
val_losses = []
val_acc = []
val_iter = []

for epoch in range(EPOCHS):

    for iter, (vectors, targets) in enumerate(trainloader):

        # Set train mode
        model.train()

        # Zero out previous gradients
        optimizer.zero_grad()

        # Predict sentiment probabilities
        probs, logits = model(vectors)

        # Compute loss and accuracy
        loss = ((probs - targets) ** 2).sum()

        # Get the predicted labels
        preds = probs.argmax(dim=1)
        targets = targets.argmax(dim=1)

        # Compute the prediction accuracy
        accuracy = (preds == targets).sum()
        accuracy = accuracy.get().float_precision()
        accuracy = 100 * (accuracy / BATCH_SIZE)

        # Backpropagate the loss
        loss.backward()

        # Update weights
        optimizer.step()

        # Decrypt the loss for logging
        loss = loss.get().float_precision()

        # get iteration number
        train_i = 1 + epoch * len(trainloader) + iter

        # append to training losses for plotting
        train_losses.append(loss.item())
        train_iter.append(train_i)
        train_acc.append(accuracy)

        # print progress in training
        print("epoch:", epoch + 1, f'\tLoss: {loss:.2f}(train)\t|\tAcc: {accuracy:.2f}%(train)', train_i)

        # Log to Tensorboard
        writer.add_scalar('train/loss', loss, train_i)
        writer.add_scalar('train/acc', accuracy, train_i)

        # break if over 100 iterations to save time
        if train_i > 100:
            break

        """ Perform validation on exactly one batch """

        # Set validation mode
        model.eval()

        for vectors, targets in valloader:
            probs, logits = model(vectors)

            loss = ((probs - targets) ** 2).sum()

            preds = probs.argmax(dim=1)
            targets = targets.argmax(dim=1)

            accuracy = preds.eq(targets).sum()
            accuracy = accuracy.get().float_precision()
            accuracy = 100 * (accuracy / BATCH_SIZE)

            # Decrypt loss for logging/plotting
            loss = loss.get().float_precision()

            # get iteration
            val_i = 1 + epoch * len(trainloader) + iter

            # append to validation losses for plotting
            val_losses.append(loss.item())
            val_iter.append(val_i)
            val_acc.append(accuracy)

            # print progress in validation
            print("epoch:", epoch + 1, f'\tLoss: {loss:.2f}(valid)\t|\tAcc: {accuracy:.2f}%(valid)', val_i)

            # Log to tensorboard
            writer.add_scalar('val/loss', loss, val_i)
            writer.add_scalar('val/acc', accuracy, val_i)

            break

writer.close()