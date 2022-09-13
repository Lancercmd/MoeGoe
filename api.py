"""
Author       : Lancercmd
Date         : 2022-08-18 19:08:42
LastEditors  : Lancercmd
LastEditTime : 2022-09-02 18:17:16
Description  : None
GitHub       : https://github.com/Lancercmd
"""
from hashlib import md5
from pathlib import Path
from re import match

from fastapi import FastAPI, Request
from scipy.io.wavfile import write
from starlette.responses import FileResponse
from torch import LongTensor
from uvicorn import run

from moegoe import utils
from moegoe.models import SynthesizerTrn
from moegoe.solver import get_text

home = Path(__file__).parent
statics = home / "statics"
models = {
    "hamidashi": ["hamidashi_604_epochs.pth", "hamidashi_config.json"],
    "yuzusoft": ["yuzusoft_365_epochs.pth", "yuzusoft_config.json"],
    "Zh_Ja": ["Zh_Ja_1374_epochs.pth", "Zh_Ja_config.json"],
}
output = home / "output"


class BaseModel:
    def __init__(self, model: str) -> None:
        pth = (statics / models[model][0]).resolve()
        cfg = (statics / models[model][1]).resolve()
        self.hps_ms = utils.get_hparams_from_file(cfg)
        self.net_g_ms = SynthesizerTrn(
            len(self.hps_ms.symbols),
            self.hps_ms.data.filter_length // 2 + 1,
            self.hps_ms.train.segment_size // self.hps_ms.data.hop_length,
            n_speakers=self.hps_ms.data.n_speakers,
            **self.hps_ms.model,
        )
        _ = self.net_g_ms.eval()
        utils.load_checkpoint(pth, self.net_g_ms)

    def synthesize(self, text: str, speaker_id: int, fp: Path, cleaned=False) -> Path:
        stn_tst = get_text(text, self.hps_ms, cleaned)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = (
            self.net_g_ms.infer(
                x_tst,
                x_tst_lengths,
                sid=sid,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=1,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        fp.parent.mkdir(parents=True, exist_ok=True)
        write(fp, self.hps_ms.data.sampling_rate, audio)
        return fp


class Hamidashi(BaseModel):
    SPEAKER_IDS = {
        "妃爱": 0,
        "华乃": 1,
        "亚澄": 2,
        "明日海": 2,
        "诗樱": 3,
        "天梨": 4,
        "里": 5,
        "里姐": 5,
        "广梦": 6,
        "莉莉子": 7,
    }
    REGEX_COMMON = r"^让(?P<speaker>" + "|".join(SPEAKER_IDS.keys()) + r")说(?P<text>.+)$"

    def __init__(self) -> None:
        self.name = "hamidashi"
        super().__init__(self.name)

    def syn(self, msg: str) -> Path:
        m = match(self.REGEX_COMMON, msg)
        if m is None:
            return None
        speaker = m.group("speaker")
        text = m.group("text")
        speaker_id = self.SPEAKER_IDS[speaker]
        fp = output / (
            f"{self.name}_{speaker_id}_"
            + md5(text.encode("utf-8")).hexdigest()
            + ".wav"
        )
        if not fp.exists():
            try:
                self.synthesize(
                    text,
                    speaker_id,
                    fp,
                )
            except IndexError:
                pass
        print(fp.name, "->", f"{speaker}:", text) if fp.is_file() else ...
        return fp


hamidashi = Hamidashi()


class Yuzusoft(BaseModel):
    SPEAKER_IDS = {
        "宁宁": 0,
        "爱瑠": 1,
        "芳乃": 2,
        "茉子": 3,
        "丛雨": 4,
        "小春": 5,
        "七海": 6,
    }
    REGEX_COMMON = r"^让(?P<speaker>" + "|".join(SPEAKER_IDS.keys()) + r")说(?P<text>.+)$"

    def __init__(self) -> None:
        self.name = "yuzusoft"
        super().__init__(self.name)

    def syn(self, msg: str) -> Path:
        m = match(self.REGEX_COMMON, msg)
        if m is None:
            return None
        speaker = m.group("speaker")
        text = m.group("text")
        speaker_id = self.SPEAKER_IDS[speaker]
        fp = output / (
            f"{self.name}_{speaker_id}_"
            + md5(text.encode("utf-8")).hexdigest()
            + ".wav"
        )
        if not fp.exists():
            try:
                self.synthesize(
                    text,
                    speaker_id,
                    fp,
                )
            except IndexError:
                pass
        print(fp.name, "->", f"{speaker}:", text) if fp.is_file() else ...
        return fp


yuzusoft = Yuzusoft()


class Zh_Ja(BaseModel):
    SPEAKER_IDS = {
        "中国宁宁": 0,
        "中国七海": 1,
        "小茸": 2,
        "唐乐吟": 3,
    }
    REGEX_COMMON = r"^让(?P<speaker>" + "|".join(SPEAKER_IDS.keys()) + r")说(?P<text>.+)$"

    def __init__(self) -> None:
        self.name = "Zh_Ja"
        super().__init__(self.name)

    def syn(self, msg: str) -> Path:
        m = match(self.REGEX_COMMON, msg)
        if m is None:
            return None
        speaker = m.group("speaker")
        text = m.group("text")
        speaker_id = self.SPEAKER_IDS[speaker]
        fp = output / (
            f"{self.name}_{speaker_id}_"
            + md5(text.encode("utf-8")).hexdigest()
            + ".wav"
        )
        if not fp.exists():
            try:
                self.synthesize(
                    text,
                    speaker_id,
                    fp,
                )
            except IndexError:
                pass
        print(fp.name, "->", f"{speaker}:", text) if fp.is_file() else ...
        return fp


zh_ja = Zh_Ja()


def syn(msg: str) -> Path:
    return hamidashi.syn(msg) or yuzusoft.syn(msg) or zh_ja.syn(msg)


def vacuum() -> None:
    from pathlib import Path
    from shutil import rmtree

    _cwd = Path.cwd()
    _pycache = _cwd.glob("**/__pycache__")
    for _path in _pycache:
        rmtree(_path)


app = FastAPI()


@app.on_event("startup")
def _() -> None:
    syn("让宁宁说こんにちは")
    syn("让爱瑠说こんにちは")
    syn("让芳乃说こんにちは")
    syn("让茉子说こんにちは")
    syn("让丛雨说こんにちは")
    syn("让小春说こんにちは")
    syn("让七海说こんにちは")

    syn("让妃爱说こんにちは")
    syn("让华乃说こんにちは")
    syn("让亚澄说こんにちは")
    syn("让诗樱说こんにちは")
    syn("让天梨说こんにちは")
    syn("让里姐说こんにちは")
    syn("让广梦说こんにちは")
    syn("让莉莉子说こんにちは")

    syn("让中国宁宁说[ZH]中文和[ZH][JA]日本語[JA][ZH]混在一起[ZH][JA]喋り[JA][ZH]的效果，还是[ZH][JA]まだまだね[JA]")
    syn("让中国七海说[ZH]中文和[ZH][JA]日本語[JA][ZH]混在一起[ZH][JA]喋り[JA][ZH]的效果，还是[ZH][JA]まだまだね[JA]")
    syn("让小茸说[ZH]中文和[ZH][JA]日本語[JA][ZH]混在一起[ZH][JA]喋り[JA][ZH]的效果，还是[ZH][JA]まだまだね[JA]")
    syn("让唐乐吟说[ZH]中文和[ZH][JA]日本語[JA][ZH]混在一起[ZH][JA]喋り[JA][ZH]的效果，还是[ZH][JA]まだまだね[JA]")


def from_local(request: Request) -> bool:
    return request.client.host == "127.0.0.1"


@app.get("/{text}")
async def _(request: Request, text: str):
    if len(text) > 100:
        return {"message": "Too many characters"}
    fp = syn(text)
    if fp and fp.is_file():
        if "local" in request.query_params and from_local(request):
            return {"path": fp}
        elif "remote" in request.query_params:
            return {"path": "/" + fp.parent.name + "/" + fp.name}
        else:
            return FileResponse(fp)
    else:
        return {"message": "No phoneme"}


@app.get("/" + output.name + "/{name}")
async def _(name: str):
    if (output / name).is_file():
        return FileResponse(output / name)
    else:
        return {"message": "Not found"}


if __name__ == "__main__":
    run(app, host="0.0.0.0", port=10721)
    vacuum()
