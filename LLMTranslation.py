import torch
import json
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

  def _split_text(self, text, spe):
    splited = text.split(spe)
    res = []
    cnt = 0
    fixedChunk = ""
    
    for item in splited:
        if (cnt + len(item) < 1024) :
            cnt += len(item)
            fixedChunk += item + spe
        else:
            res.append(fixedChunk)
            cnt=len(item)
            fixedChunk = item + spe
    return res
          
  def translateText(self, text, src, tgt):
    if self._isAvailable(src) and self._isAvailable(tgt):
        return self._translate(text, src, tgt)
    else:
        print("[ERROR] check languages..")
        return ""

  def translateJson(self, json_file, src, tgt):
    with open(json_file, "r") as read_file:
        data = json.load(read_file)
    trans_lists = ['title', 'topic', 'questions', 'context', 'summary']  
    for item in data:
        for key in item.keys():
            print("LOG item.keys()", key)
            if key in trans_lists:
                if isinstance(item[key],str) == True:
                    if (len(item[key]) <= 1024):
                        translated_text = self._translate(item[key], src, tgt)
                        item[key] = translated_text
                    else:
                        sep = "\n" if key == 'context' else "."
                        chunks = self._split_text(item[key], sep)
                        translated_chunks = ""
                        for chunk in chunks:
                            translated_text = self._translate(chunk, src, tgt)
                            translated_chunks += translated_text
                        item[key] = translated_chunks
                
                elif key == 'questions':
                    ques_keys = list(item[key].keys())
                    output = {}
                    for idx_i, ques_key in enumerate(ques_keys):
                        if (len(ques_keys[idx_i]) <= 1024):
                            translated_ques = self._translate(ques_keys[idx_i], src, tgt)
                            answers = item['questions'][ques_keys[idx_i]]
                            for idx_j, ans in enumerate(answers):
                                translated_ans = self._translate(ans['answer'], src, tgt)
                                item['questions'][ques_keys[idx_i]][idx_j]['answer'] = str(translated_ans)
                        output[str(translated_ques)] = list(item['questions'][ques_keys[idx_i]])
                    #replace questions to translated question, because question_text is the key of dictionary
                    item['questions'] = output

    file_name = "tr_result_" + src +"_"+tgt+".json"
    with open(file_name, "w") as f:
        json.dump(data, f)
      
    return data


