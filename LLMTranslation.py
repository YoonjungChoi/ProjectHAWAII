import torch
from seamless_communication.models.inference import Translator

class SeamlessMT4:

  def __init__(self, device=torch.device("cpu")):
      print("[LOG] HFSeamlessM4T init...")
      self._translator = Translator("seamlessM4T_medium", vocoder_name_or_card="vocoder_36langs", device=device)
      #check supported languages - https://github.com/facebookresearch/seamless_communication/tree/main/scripts/m4t/predict
      self._langs = ['afr', 'amh', 'arb', 'ary', 'arz', 'asm', 'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ceb', 'ces', 'ckb', 'cmn', 'cmn_Hant', 'cym',
                    'dan', 'deu', 'ell', 'eng', 'est', 'eus', 'fin', 'fra', 'gaz', 'gle', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ibo', 'ind', 
                     'isl', 'ita', 'jav', 'jpn', 'kan', 'kat', 'kaz', 'khk', 'khm', 'kir', 'kor', 'lao', 'lit', 'lug', 'luo', 'lvs', 'mai', 'mal',
                     'mar', 'mkd', 'mlt', 'mni', 'mya', 'nld', 'nno', 'nob', 'npi', 'nya', 'ory', 'pan', 'pbt', 'pes', 'pol', 'por', 'ron', 'rus', 'slk',
                     'slv', 'sna', 'snd', 'som', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgk', 'tgl', 'tha', 'tur', 'ukr', 'urd', 'uzn', 'vie', 'yor',
                     'yue', 'zsm', 'zul']

  def _translate(self, text, src, tgt):
    result, _, _ = self._translator.predict(text, "t2tt", tgt, src_lang=src)
    return str(result)

  def _isAvailable(self, lang):
      if lang in self._langs:
          return True
      return False
          
  def translateText(self, text, src, tgt):
    if self._isAvailable(src) and self._isAvailable(tgt):
        return self._translate(text, src, tgt)
    else:
        print("[ERROR] check languages..")
        return ""

  def translateJson(self, json_text, src, tgt):
    translated_json_text = ""
    return translated_json_text


