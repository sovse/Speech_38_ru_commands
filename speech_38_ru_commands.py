import torch
import torch.nn as nn
import torchaudio
from conformer import ConformerBlock
from typing import List
import sounddevice as sd
import numpy as np
import torch.nn.functional as F
import sys
import warnings
import argparse
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Список классов для предсказаний
parser = argparse.ArgumentParser()
parser.add_argument("--porog", type=float, default=1.2)
args = parser.parse_args()
porog=args.porog
CLASSES = [
    'дальше',
    'вперед',
    'назад',
    'вверх',
    'вниз',
    'выше',
    'ниже',
    'домой',
    'громче',
    'тише',
    'лайк',
    'дизлайк',
    'следующий',
    'предыдущий',
    'сначала',
    'перемотай',
    'выключи',
    'стоп',
    'хватит',
    'замолчи',
    'заткнись',
    'останови',
    'пауза',
    'включи',
    'смотреть',
    'продолжи',
    'играй',
    'запусти',
    'ноль',
    'один',
    'два',
    'три',
    'четыре',
    'пять',
    'шесть',
    'семь',
    'восемь',
    'девять'
]
MAX_LEN=321
hparams = {
        "n_cnn_layers": 2,
        "n_rnn_layers": 2,
        "rnn_dim": 512,
        "n_class": 28,
        "n_feats": 80,
        "stride":2,
        "dropout": 0.1,
}
# Hyper-parameters
# sequence_length = 28
input_size = 40
hidden_size = 512
num_layers = 3
num_classes = len(CLASSES)


# bluid models
class Conv1dModule(nn.Module):
    """
    Простой кирпичик для свёрточной модели.
    n_in -- число фильтров на входе
    n_out -- число фильтров на выходе
    kernel -- размер ядра
    pooling -- размер ядра пулинга
    batchnorm -- флаг отвечающий за использование батч нормализации
    relu -- флаг отвечающий за использование нелинейности
    """
    def __init__(self, n_in, n_out, kernel, pooling, batchnorm=False, relu=True):
        super().__init__()
        assert kernel % 2 == 1
        pad = kernel // 2
        modules = [nn.Conv1d(n_in, n_out, kernel, padding=pad)]
        if batchnorm:
            modules.append(nn.BatchNorm1d(n_out))
        if pooling > 1:
            modules.append(nn.MaxPool1d(pooling))
        if relu:
            modules.append(nn.ReLU())
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)


class Flatten(nn.Module):
    def forward(self, X):
        return X.reshape(X.shape[0], -1)


class FlattenModule(nn.Module):
    """
    Простой кирпичик полносвязной части свёрточной модели.
    n_in -- число нейронов на входе
    n_out -- число нейроново на выходе
    batchnorm -- флаг отвечающий за использование батч нормализации
    relu -- флаг отвечающий за использование нелинейности
    """
    def __init__(self, n_in, n_out, batchnorm=False, relu=True):
        super().__init__()
        modules = [nn.Linear(n_in, n_out)]
        if batchnorm:
            modules.append(nn.BatchNorm1d(n_out))
        if relu:
            modules.append(nn.ReLU())
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)

block = ConformerBlock(
    dim = 40,
    dim_head = 64,
    heads = 8,
    ff_mult = 4,
    conv_expansion_factor = 2,
    conv_kernel_size = 31,
    attn_dropout = 0.,
    ff_dropout = 0.,
    conv_dropout = 0.
    )
class Conv1dModel(nn.Module):
    """
    Пример простой сети для работы со звуком.
    Изначально несколько слоёв используют 1d свёртки, затем просходит конкатенация матрицы
    признаков в вектор и применяются несколько полносвязных слоёв.
    В целом тут всё очень сильно похоже на картинки.

    shapes -- число филтров в свёрточных слоях. На нулевом индексе число фильтров на входе.
        Далее число фильтров после каждого свёрточного слоя
    flatten_shapes -- число нейронов после применения линейных слоёв.
    kernels -- размеры ядер свёрточных слоёв
    poolings -- параметры пулинга после свёрточных слоёв
    batchnorm -- флаг отвечающий за использовать или нет нормализацию
    """
    def __init__(
        self, shapes: List[int], flatten_shapes: List[int], kernels: List[int],
        poolings: List[int], batchnorm=False
    ):
        super().__init__()
        assert len(kernels) + 1 == len(shapes)
        assert len(poolings) == len(kernels)
        modules = []
        start_flatten_shape = MAX_LEN
        for i in range(len(kernels)):
            modules.append(Conv1dModule(
                shapes[i], shapes[i + 1], kernels[i], poolings[i],
                batchnorm=batchnorm
            ))
            start_flatten_shape //= poolings[i]
        modules.append(block)
        modules.append(Flatten())
        flatten_shapes = [start_flatten_shape * shapes[-1]] + flatten_shapes
        for i in range(len(flatten_shapes) - 1):
            modules.append(FlattenModule(
                flatten_shapes[i], flatten_shapes[i + 1], batchnorm=batchnorm, relu=i+2==len(flatten_shapes)
            ))
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)

class SimpleDenoiserModel(nn.Module):
    """
    Очень простой денойзер состоящий из маленькой свёрточной модели.
    shapes -- число фильтров в свёрточных слоях. На нулевом индексе число фильтров на входе
    kernels -- размерность ядер в свёрточных слоях
    batchnorm -- флаг отвечающий за использование нормализации
    """
    def __init__(self, shapes: List[int], kernels: List[int], batchnorm=False):
        super().__init__()
        assert len(shapes) == len(kernels) + 1
        modules = []
        for i in range(len(kernels)):
            modules.append(Conv1dModule(
                shapes[i], shapes[i + 1], kernels[i], 1, batchnorm=batchnorm, relu=i + 1!=len(kernels)
            ))
        self._net = nn.Sequential(*modules)

    def forward(self, X):
        return self._net.forward(X)
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)


class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x


class SpeechRecognitionModel(nn.Module):

    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )
        self.lstm0 = nn.GRU(rnn_dim*2, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, len(CLASSES))
    def forward(self, x):

        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x0 = self.birnn_layers(x)
        out,_=self.lstm0(x0)
        out = self.fc(out[:, -1, :])
        x = self.classifier(x0)
        return x,out
class DueModel(nn.Module):
    def __init__(self):
        super(DueModel, self).__init__()
        self.conv1d_clean_model= Conv1dModel( [80] + [256] * 7, [1024, 256, len(CLASSES)],[5] * 7, [1, 2] * 3 + [1], batchnorm=True)
        self.denoser = SimpleDenoiserModel([40, 64, 128,128, 64, 40], [3] * 5)

        self.dp2 = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout'])
        self.ob=nn.Linear(len(CLASSES)*2, len(CLASSES))
#         self.ob1=nn.Linear(hparams['rnn_dim'], hparams['n_class'])
    def forward(self, x):
        x1 = self.denoser(x)


        x2= torch.cat((x1, x), dim=1)
        x21=x2.unsqueeze(1)
        x4,x5=self.dp2(x21)
        x3 = self.conv1d_clean_model(x2)
        x6= torch.cat((x3, x5), dim=-1)
        x7=self.ob(x6)
#         x7=self.ob1(x7)
        return x7,x3,x5,x1 ,x4

model = DueModel().to(device)
model.load_state_dict(torch.load('model.0'))
model.eval()

# цикл распознавания
prepare_fun=torchaudio.transforms.MFCC(melkwargs={'n_mels': 80})
duration =0.5 # seconds
fs=16000
k=0
old=''
z=0
sys.stdout.write('Произнеси в микрофон слово из списка: \n')
for i in  CLASSES:
    sys.stdout.write(i+', ')
sys.stdout.write('\n==============================================\n' )
sys.stdout.write('слово   -   уверенность\n' )
while True:
    if z==0:
        a1 = sd.rec(int(4 * fs), samplerate=fs, channels=1,dtype=np.float32)
        sd.wait()
        z=2
    else:
        a1[:-8000,:]=a1[8000:,:]
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1,dtype=np.float32)
        sd.wait()
        a1[-8000:,:]=myrecording
    b=np.moveaxis(a1, 0, -1)
    audio=torch.from_numpy(b)
    data=prepare_fun(audio)
    data=data.to(device)
    a,_,_,_,_=model.forward(data)
    preds=a.cpu()
    probs=torch.softmax(preds, dim=-1)
    probas, classes = torch.max(probs, dim=-1)
    # classes = torch.argmax(a, dim=-1).cpu().data.numpy()
    labels = [CLASSES[idx] for idx in classes]
    zz=probas.tolist()[0]
    if labels[0]==old:
        k+=zz
    else:
        k=zz
    old=labels[0]
    if k>porog:


        sys.stdout.write('\r'+str(labels[0])+'  -  '+str(k))
        sys.stdout.flush()
#         print(labels[0],k)
