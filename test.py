from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from pathlib import Path

curr_dir = Path(__file__).parent

model_dir = "./SenseVoiceSmall"
vad_model_dir = "./speech_fsmn_vad_zh-cn-16k-common-pytorch"

model = AutoModel(
    model=model_dir,
    # vad_model="fsmn-vad",
    vad_model=vad_model_dir,
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=f"E:\Code\PythonDir\MCP\mcp-server-funasr\Data\_20240821153822.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
# print(res)
print(text)