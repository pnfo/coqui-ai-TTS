import os

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    formatter="pathnirvana2", meta_file_train="metadata_shuf.csv", path=os.path.join(output_path, "pali_dataset/")
)
audio_config = VitsAudioConfig(
    sample_rate=22050, 
    win_length=1024, 
    hop_length=256, 
    num_mels=80, 
    mel_fmin=0,
    mel_fmax=None
)
vitsArgs = VitsArgs(
    use_speaker_embedding=True,
    #num_speakers=2 # can set here or later using the speaker_manager
)
config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_pali_speaker",
    use_speaker_embedding=True,
    batch_size=54,
    eval_batch_size=32,
    batch_group_size=5,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner=None, # the text prompts were is already cleaned, remove extra whitespace, special symbols etc
    use_phonemes=False,
    #phoneme_language="en-us",
    #phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    max_audio_len=15 * 22050, # audio longer than this will be ignored
    add_blank=True, # this is by default true for vits, not sure if needed, speed is not changed by much
    #enable_eos_bos_chars=True, # causes None char to showup in the tokenizer ;-)
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=" '(),-.:;?abcdeghijklmnoprstuvxyzñāīūḍḷṃṅṇṭ",
        punctuations=" '(),-.:;?xz",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    test_sentences=[
        ["suppiyassa pana paribbājakassa antevāsī brahmadatto māṇavo anekapariyāyena buddhassa vaṇṇaṃ bhāsati, dhammassa vaṇṇaṃ bhāsati, saṅghassa vaṇṇaṃ bhāsati.", "mettananda", None, None],
        ["namo tassa bhagavato arahato sammā sambuddhassa", "obhasa", None, None],
        ["manopubbaṅgamā dhammā manoseṭṭhā manomayā x manasā ce paduṭṭhena bhāsati vā karoti vā x tato naṃ dukkhamanveti, cakkaṃ'va vahato padaṃ.", "oshadir", None, None],
        ["mālāgandhavilepanadhāraṇamaṇḍanavibhūsanaṭṭhānā veramaṇīsikkhāpadaṃ samādiyāmi.", "mettananda", None, None],
        ["sekhabalasaṅkhittasuttaṃz", "oshadir", None, None],
        ["yo brāhmaṇo bāhitapāpadhammo x nihuhuṅko nikkasāvo yatatto x vedantagū vusitabrahmacariyo", "wdevananda", None, None],
        ["kittāvatā saccānaṃ saccapaññatti: yāvatā cattāri saccāni, dukkhasaccaṃ samudayasaccaṃ nirodhasaccaṃ maggasaccaṃ. ettāvatā saccānaṃ saccapaññatti.", "wdevananda", None, None],
        #["සුප්පියස්ස පන පරිබ්බාජකස්ස අන්තෙවාසී බ්රහ්මදත්තො මාණවො අනෙකපරියායෙන බුද්ධස්ස වණ්ණං භාසති, ධම්මස්ස වණ්ණං භාසති, සඞ්ඝස්ස වණ්ණං භාසති."],
        #["නමො තස්ස භගවතො අරහතො සම්මා සම්බුද්ධස්ස"],
        #["මනොපුබ්බඞ්ගමා ධම්මා මනොසෙට්ඨා මනොමයා x මනසා චෙ පදුට්ඨෙන භාසති වා කරොති වා x තතො නං දුක්ඛමන්වෙති, චක්කං'ව වහතො පදං."],
        #["මාලාගන්ධවිලෙපනධාරණමණ්ඩනවිභූසනට්ඨානා වෙරමණීසික්ඛාපදං සමාදියාමි."],
        #["සෙඛබලසඞ්ඛිත්තසුත්තං"],
        #["යො බ්‍රාහ්මණො බාහිතපාපධම්මො x නිහුහුඞ්කො නික්කසාවො යතත්තො x වෙදන්තගූ වුසිතබ්‍රහ්මචරියො"],
        #["කිත්තාවතා සච්චානං සච්චපඤ්ඤත්ති: යාවතා චත්තාරි සච්චානි, දුක්ඛසච්චං සමුදයසච්චං නිරොධසච්චං මග්ගසච්චං. එත්තාවතා සච්චානං සච්චපඤ්ඤත්ති."],
    ],
    print_step=50,
    print_eval=False,
    mixed_precision=True, # try with false since other multilanguage training was done like that
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    eval_split_max_size=300, # max number of eval samples 
    eval_split_size=0.1, # 10% of the samples to eval
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers
print("Num Speakers: ", speaker_manager.num_speakers)

# init model
model = Vits(config, ap, tokenizer, speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
