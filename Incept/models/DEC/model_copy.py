import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Optional, Tuple

def default_initialise_weight_bias_(
    weight: torch.Tensor, bias: torch.Tensor, gain: float
) -> None:
    """
    Default function to initialise the weights in a the Linear units of the StackedDenoisingAutoEncoder.

    :param weight: weight Tensor of the Linear unit
    :param bias: bias Tensor of the Linear unit
    :param gain: gain for use in initialiser
    :return: None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)

def build_units(
    dimensions: Iterable[int], activation: Optional[torch.nn.Module]
) -> List[torch.nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear
    layer followed by an activation layer.

    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """
    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [("linear", nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(("activation", activation))
        return nn.Sequential(OrderedDict(unit))

    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension in sliding_window(2, dimensions)
    ]

class DenoisingAutoencoder(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: int,
        activation: Optional[torch.nn.Module] = nn.ReLU(),
        gain: float = nn.init.calculate_gain("relu"),
        corruption: Optional[torch.nn.Module] = None,
        tied: bool = False,
    ) -> None:
        """
        Autoencoder composed of two Linear units with optional encoder activation and corruption.

        :param embedding_dimension: embedding dimension, input to the encoder
        :param hidden_dimension: hidden dimension, output of the encoder
        :param activation: optional activation unit, defaults to nn.ReLU()
        :param gain: gain for use in weight initialisation
        :param corruption: optional unit to apply to corrupt input during training, defaults to None
        :param tied: whether the autoencoder weights are tied, defaults to False
        """
        super(DenoisingAutoencoder, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension
        self.activation = activation
        self.gain = gain
        self.corruption = corruption
        # encoder parameters
        self.encoder_weight = Parameter(
            torch.Tensor(hidden_dimension, embedding_dimension)
        )
        self.encoder_bias = Parameter(torch.Tensor(hidden_dimension))
        self._initialise_weight_bias(self.encoder_weight, self.encoder_bias, self.gain)
        # decoder parameters
        self._decoder_weight = (
            Parameter(torch.Tensor(embedding_dimension, hidden_dimension))
            if not tied
            else None
        )
        self.decoder_bias = Parameter(torch.Tensor(embedding_dimension))
        self._initialise_weight_bias(self._decoder_weight, self.decoder_bias, self.gain)

    @property
    def decoder_weight(self):
        return (
            self._decoder_weight
            if self._decoder_weight is not None
            else self.encoder_weight.t()
        )

    @staticmethod
    def _initialise_weight_bias(weight: torch.Tensor, bias: torch.Tensor, gain: float):
        """
        Initialise the weights in a the Linear layers of the DenoisingAutoencoder.

        :param weight: weight Tensor of the Linear layer
        :param bias: bias Tensor of the Linear layer
        :param gain: gain for use in initialiser
        :return: None
        """
        if weight is not None:
            nn.init.xavier_uniform_(weight, gain)
        nn.init.constant_(bias, 0)

    def copy_weights(self, encoder: torch.nn.Linear, decoder: torch.nn.Linear) -> None:
        """
        Utility method to copy the weights of self into the given encoder and decoder, where
        encoder and decoder should be instances of torch.nn.Linear.

        :param encoder: encoder Linear unit
        :param decoder: decoder Linear unit
        :return: None
        """
        encoder.weight.data.copy_(self.encoder_weight)
        encoder.bias.data.copy_(self.encoder_bias)
        decoder.weight.data.copy_(self.decoder_weight)
        decoder.bias.data.copy_(self.decoder_bias)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        transformed = F.linear(batch, self.encoder_weight, self.encoder_bias)
        if self.activation is not None:
            transformed = self.activation(transformed)
        if self.corruption is not None:
            transformed = self.corruption(transformed)
        return transformed

    def decode(self, batch: torch.Tensor) -> torch.Tensor:
        return F.linear(batch, self.decoder_weight, self.decoder_bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))

class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
        self,
        dimensions: List[int],
        activation: torch.nn.Module = nn.ReLU(),
        final_activation: Optional[torch.nn.Module] = nn.ReLU(),
        weight_init: Callable[
            [torch.Tensor, torch.Tensor, float], None
        ] = default_initialise_weight_bias_,
        gain: float = nn.init.calculate_gain("relu"),
    ):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimensions input is the list of dimensions occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].

        :param dimensions: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        :param weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        :param gain: gain parameter to pass to weight_init
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        # construct the encoder
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(
            build_units([self.dimensions[-2], self.dimensions[-1]], None)
        )
        self.encoder = nn.Sequential(*encoder_units)
        # construct the decoder
        decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(
            build_units([self.dimensions[1], self.dimensions[0]], final_activation)
        )
        self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.

        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError(
                "Requested subautoencoder cannot be constructed, index out of range."
            )
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)
        return self.decoder(encoded)

class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

class DECModel(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DECModel, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        return self.assignment(self.encoder(batch))

class DEC:
    def __init__(self, config):
        self.config = config

    def pretrain(
        self,
        dataset,
        autoencoder: StackedDenoisingAutoEncoder,
        epochs: int,
        batch_size: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
        validation: Optional[torch.utils.data.Dataset] = None,
        corruption: Optional[float] = None,
        cuda: bool = True,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        silent: bool = False,
        update_freq: Optional[int] = 1,
        update_callback: Optional[Callable[[float, float], None]] = None,
        num_workers: Optional[int] = None,
        epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    ):
        current_dataset = dataset
        current_validation = validation
        number_of_subautoencoders = len(autoencoder.dimensions) - 1
        for index in range(number_of_subautoencoders):
            encoder, decoder = autoencoder.get_stack(index)
            embedding_dimension = autoencoder.dimensions[index]
            hidden_dimension = autoencoder.dimensions[index + 1]
            # manual override to prevent corruption for the last subautoencoder
            if index == (number_of_subautoencoders - 1):
                corruption = None
            # initialise the subautoencoder
            sub_autoencoder = DenoisingAutoencoder(
                embedding_dimension=embedding_dimension,
                hidden_dimension=hidden_dimension,
                activation=torch.nn.ReLU()
                if index != (number_of_subautoencoders - 1)
                else None,
                corruption=nn.Dropout(corruption) if corruption is not None else None,
            )
            if cuda:
                sub_autoencoder = sub_autoencoder.cuda()
            ae_optimizer = optimizer(sub_autoencoder)
            ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler

            self.train(
                current_dataset,
                sub_autoencoder,
                epochs,
                batch_size,
                ae_optimizer,
                validation=current_validation,
                corruption=None,  # already have dropout in the DAE
                scheduler=ae_scheduler,
                cuda=cuda,
                sampler=sampler,
                silent=silent,
                update_freq=update_freq,
                update_callback=update_callback,
                num_workers=num_workers,
                epoch_callback=epoch_callback,
            )

            sub_autoencoder.copy_weights(encoder, decoder)
            if index != (number_of_subautoencoders - 1):
                current_dataset = TensorDataset(
                    self.predict(
                        current_dataset,
                        sub_autoencoder,
                        batch_size,
                        cuda=cuda,
                        silent=silent,
                    )
                )

                if current_validation is not None:
                    current_validation = TensorDataset(
                        self.predict(
                            current_validation,
                            sub_autoencoder,
                            batch_size,
                            cuda=cuda,
                            silent=silent,
                        )
                    )
            else:
                current_dataset = None  # minor optimisation on the last subautoencoder
                current_validation = None

        # print(self.autoencoder)

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        autoencoder: torch.nn.Module,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        validation: Optional[torch.utils.data.Dataset] = None,
        corruption: Optional[float] = None,
        cuda: bool = True,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        silent: bool = False,
        update_freq: Optional[int] = 1,
        update_callback: Optional[Callable[[float, float], None]] = None,
        num_workers: Optional[int] = None,
        epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    ):
        # self.pretrain()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=True if sampler is None else False,
            num_workers=num_workers if num_workers is not None else 0,
        )
        if validation is not None:
            validation_loader = DataLoader(
                validation,
                batch_size=batch_size,
                pin_memory=False,
                sampler=None,
                shuffle=False,
            )
        else:
            validation_loader = None

        loss_function = nn.MSELoss()
        autoencoder.train()
        validation_loss_value = -1
        loss_value = 0

        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()

            data_iterator = tqdm(
                dataloader,
                leave=True,
                unit="batch",
                postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1,},
                disable=silent,
            )

            for index, batch in enumerate(data_iterator):
                if (
                    isinstance(batch, tuple)
                    or isinstance(batch, list)
                    and len(batch) in [1, 2]
                ):
                    batch = batch[0]
                    if cuda:
                        batch = batch.cuda(non_blocking=True)
                    
                    if corruption is not None:
                        output = autoencoder(F.dropout(batch, corruption))
                    else:
                        output = autoencoder(batch)
                                        
                    loss = loss_function(output, batch)
                    loss_value = float(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(closure=None)
                    data_iterator.set_postfix(
                        epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
                    )
                
                if update_freq is not None and epoch % update_freq == 0:
                    if validation_loader is not None:
                        validation_output = self.predict(
                            validation,
                            autoencoder,
                            batch_size,
                            cuda=cuda,
                            silent=True,
                            encode=False,
                        )

                        validation_inputs = []
                        for val_batch in validation_loader:
                            if (
                                isinstance(val_batch, tuple) or isinstance(val_batch, list)
                            ) and len(val_batch) in [1, 2]:
                                validation_inputs.append(val_batch[0])
                            else:
                                validation_inputs.append(val_batch)
                        validation_actual = torch.cat(validation_inputs)
                        if cuda:
                            validation_actual = validation_actual.cuda(non_blocking=True)
                            validation_output = validation_output.cuda(non_blocking=True)
                        validation_loss = loss_function(validation_output, validation_actual)
                        # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                        validation_loss_value = float(validation_loss.item())
                        data_iterator.set_postfix(
                            epo=epoch,
                            lss="%.6f" % loss_value,
                            vls="%.6f" % validation_loss_value,
                        )
                        autoencoder.train()
                    else:
                        validation_loss_value = -1
                        # validation_accuracy = -1
                        data_iterator.set_postfix(
                            epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                        )
                    
                    if update_callback is not None:
                        update_callback(
                            epoch,
                            optimizer.param_groups[0]["lr"],
                            loss_value,
                            validation_loss_value,
                        )
            if epoch_callback is not None:
                autoencoder.eval()
                epoch_callback(epoch, autoencoder)
                autoencoder.train()
        
    def predict(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        batch_size: int,
        cuda: bool = True,
        silent: bool = False,
        encode: bool = True,
    ):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
        features = []
        if isinstance(model, torch.nn.Module):
            model.eval()
        for batch in data_iterator:
            if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            batch = batch.squeeze(1).view(batch.size(0), -1)
            if encode:
                output = model.encode(batch)
            else:
                output = model(batch)
            features.append(
                output.detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU
        return torch.cat(features)

class CachedMNIST(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST("/data2/liangguanbao/opendeepclustering/datasets", download=True, train=train, transform=img_transform)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        return (
            torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()
            * 0.02
        )

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = torch.tensor(
                    self._cache[index][1], dtype=torch.long
                ).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return 128 if self.testing_mode else len(self.ds)


import sys
sys.path.append("/data2/liangguanbao/opendeepclustering/Incept")
from Incept.utils import load_config, seed_everything

seed_everything(42)

config = load_config("/data2/liangguanbao/opendeepclustering/Incept/Incept/configs/DEC/DEC_Mnist.yaml")
model = DEC(config)

ds_train = CachedMNIST(
    train=True, cuda=config.device, testing_mode=False
)

ds_val = CachedMNIST(
    train=False, cuda=config.device, testing_mode=False
)

autoencoder = StackedDenoisingAutoEncoder(
    [28 * 28, 500, 500, 2000, 10], final_activation=None
).to(config.device)
# print(autoencoder.encoder)
# print(autoencoder.encoder[0][0].weight.data)
model.pretrain(
    ds_train,
    autoencoder,
    cuda=config.device,
    validation=ds_val,
    epochs=config.pretrain_epochs,
    batch_size=config.batch_size,
    optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
    scheduler=lambda x: StepLR(x, 100, gamma=0.1),
    corruption=0.2,
)

ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
model.train(
    ds_train,
    autoencoder,
    cuda=config.device,
    validation=ds_val,
    epochs=config.finetune_epochs,
    batch_size=config.batch_size,
    optimizer=ae_optimizer,
    scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
    corruption=0.2,
    update_callback=None,
)

# print(autoencoder)

# print(ds_train[0][0][300:350])

# print(model.config)
# model.train()