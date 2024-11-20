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
    formatter="pathnirvana2", meta_file_train="metadata_shuf.csv", path=os.path.join(output_path, "sinhala_dataset/")
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
    run_name="vits_sinhala_speaker",
    use_speaker_embedding=True,
    batch_size=56,
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
        characters=" !'(),-.:;=?abcdefghijklmnoprstuvyæñāēīōśşūǣḍḥḷṁṅṇṉṛṝṭ",
        punctuations=" !'(),-.:;=?",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    test_sentences=[
        ["namo tassa bhagavato arahato sammā sambuddhassa", "mettananda", None, None],
        ["manopubbaṅgamā dhammā manoseṭṭhā manomayā ... manasā ce paduṭṭhena bhāsati vā karoti vā ... tato naṁ dukkhamanveti, cakkaṁ'va vahato padaṁ.", "oshadi", None, None],
        ["mālāgandhavilepanadhāraṇamaṇḍanavibhūsanaṭṭhānā akşara vinyāsaya ho naṇalaḷa bhedaya nosalakā sevīma kaḷa yutu nam iṁgrīsi akuru yodāganna", "mettananda", None, None],
        ["ehi dī suppiya pirivæji noyek karuṇin budurajāṇan vahansēṭa dos kiyayi,", "mettananda", None, None],
        ["ikbiti rǣ aluyamhi nægī siṭi, nişīdana śālāyehi ræs væ hun bohō bhikşūn ataræ mē kathāva pahaḷa viya:", "mettananda", None, None],
        ["no hētaṁ bhantē.", "mettananda", None, None],
        ["'luṇu jivhāviññeyya dæ?' yi kīha. 'ese ya, maharajāneni, divaṭa dænenne ya' yi kīseka. 'svāmīni, kimek da?'", "oshadi", None, None], 
        ["idin piḷikul dæyehit nopiḷikul dæyehit piḷikul saṁgnāva æti væ vesem vā yi kæmæti vē da ehi piḷikul saṁgnāva ætivæ veseyi.", "mettananda", None, None], # in dataset
        #["එහි දී සුප්පිය පිරිවැජි නොයෙක් කරුණින් බුදුරජාණන් වහන්සේට දොස් කියයි,"],
        #["ඉක්බිති රෑ අලුයම්හි නැගී සිටි, නිෂීදන ශාලායෙහි රැස් වැ හුන් බොහෝ භික්‍ෂූන් අතරැ මේ කථාව පහළ විය:"],
        #["නො හෙතං භන්තෙ."],
        #["එය මෙතෙකැයි ප්රමාණ කරන්නට ද නුපුළුවන."], # in dataset
        #["ඉදින් පිළිකුල් දැයෙහිත් නොපිළිකුල් දැයෙහිත් පිළිකුල් සංඥාව ඇති වැ වෙසෙම් වා යි කැමැති වේ ද එහි පිළිකුල් සංඥාව ඇතිවැ වෙසෙයි."], # in dataset
    ],
    print_step=50,
    print_eval=False,
    mixed_precision=True, # try with false since other multilanguage training was done like that
    output_path=output_path,
    datasets=[dataset_config],
    cudnn_benchmark=False,
    eval_split_max_size=200, # max number of eval samples 
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
