import torch
from seamless_communication.models.inference import Translator

class HFSeamlessM4T:
  def __init__(self, device=torch.device("cpu")):
      print("[LOG] HFSeamlessM4T init...")
      _translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=torch.device(device))

  def _translate(self, text, src, tgt):
    result, _, _ = translator.predict(input_text, "t2tt", tgt, src_lang=src)
    return result

  def translateText(self, text, src, tgt):
    return self._translate(text, src, tgt)

  def translateJson(json_text, src, tgt):
    translated_json_text = ""
    return translated_json_text


