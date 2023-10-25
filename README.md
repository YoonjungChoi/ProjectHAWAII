# How to use "LLMTranslation.py"


```
import torch
import LLMTranslation

myTranslator = LLMTranslation.SeamlessMT4(device=torch.device("cuda:0"))
result = myTranslator.translateText("How are you, Today? It's a sunny day!", "eng", "tgl")

#'Kumusta ka na, ngayon ay isang araw na may araw!'
```



