import warnings
warnings.filterwarnings("ignore")
import torch
# import torch.nn as nn
import torchaudio
import onnxruntime as ort
# from conformer import ConformerBlock
# from typing import List
import sounddevice as sd
import numpy as np
# import torch.nn.functional as F
import sys

import argparse
# from torchvision import  models


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

ort_session = ort.InferenceSession("test2.onnx")
MAX_LEN=321
MAX_LEN1=MAX_LEN-2


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
X1 = np.zeros((1,3,40,319),dtype=np.float32)
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
    # print(b.shape,';;;;;;;;;;;;;;;;')
    audio=torch.from_numpy(b)
    X=prepare_fun(audio)
    X=X.cpu().detach().numpy()
    X1[:,0,:,:]=X[:,:,:-2]
    X1[:,1,:,:]=X[:,:,1:-1]
    X1[:,2,:,:]=X[:,:,2:]
    outputs = ort_session.run(  None, {"input1": X1},  )
    # data=X1.to(device)
    # a=model.forward(data)
    preds=torch.from_numpy(outputs[0])

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



        sys.stdout.write('\r'+str(labels[0].upper())+'  -  '+str(k))


        sys.stdout.flush()
#         print(labels[0],k)
