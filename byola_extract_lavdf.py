from byol_a.common import *
from byol_a.augmentations import PrecomputedNorm
from byol_a.models import AudioNTT2020Feature
from sklearn.decomposition import PCA
import torchaudio.transforms as T
import librosa
import tqdm



class MelSpectrogramLibrosa:
    """Mel spectrogram using librosa."""
    def __init__(self, fs=16000, n_fft=1024, shift=160, n_mels=64, fmin=60, fmax=7800):
        self.fs, self.n_fft, self.shift, self.n_mels, self.fmin, self.fmax = fs, n_fft, shift, n_mels, fmin, fmax
        self.mfb = librosa.filters.mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def __call__(self, audio):
        X = librosa.stft(np.array(audio), n_fft=self.n_fft, hop_length=self.shift)
        return torch.tensor(np.matmul(self.mfb, np.abs(X)**2 + np.finfo(float).eps))

def calc_norm_stats(cfg, file_list, n_stats=10000):
    unit_length = int(cfg.unit_sec * cfg.sample_rate)
    to_melspec = MelSpectrogramLibrosa(
            fs=cfg.sample_rate,
            n_fft=cfg.n_fft,
            shift=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
        )
    res = []
    for file in file_list:
        wav, sr = torchaudio.load(file.strip())
        if wav.shape[0] > 1:
            wav = np.mean(wav, axis=0)
        if sr != cfg.sample_rate:
            resampler = T.Resample(sr, cfg.sample_rate, dtype=wav.dtype)
            wav = resampler(wav)
            # print(f'Convert .wav files to {cfg.sample_rate} Hz.')
        # assert wav.shape[0] == 1, f'Convert .wav files to single channel audio, {self.files[idx]} has {wav.shape[0]} channels.'
        wav = wav[0] # (1, length) -> (length,)

        # zero padding to both ends
        length_adj = unit_length - len(wav)
        if length_adj > 0:
            half_adj = length_adj // 2
            wav = F.pad(wav, (half_adj, length_adj - half_adj))

        # random crop unit length wave
        length_adj = len(wav) - unit_length
        start = random.randint(0, length_adj) if length_adj > 0 else 0
        wav = wav[start:start + unit_length]

        # to log mel spectrogram -> (1, n_mels, time)
        lms = (to_melspec(wav) + torch.finfo().eps).log().unsqueeze(0)

        # transform (augment)
        # if self.tfms:
        #     lms = self.tfms(lms)

        # if self.labels is not None:
        #     return lms, torch.tensor(self.labels[idx])
        res.append(lms)
    res = np.hstack(res)
    norm_stats = np.array([res.mean(), res.std()])
    return norm_stats

# ** Prepare the statistics in advance **
# You need to calculate the statistics of mean and standard deviation of the log-mel spectrogram of your dataset.
# See calc_norm_stats in evaluate.py for your reference.

def run(cfg_path, splits, output_dataset_name='byola', pca_dim=None, pca_fit_samples=10000):
    device = torch.device('cuda')
    cfg = load_yaml_config(cfg_path)
    print(cfg)
    
    #splits = ['dev','train']
    for split in splits:
        file_list_path = cfg.path_prefix + 'data/lavdf/video/audiofilelist/{}.txt'.format(split)
        output_file_path = cfg.path_prefix + '../UMMAFormerTest/data/lavdf/feats/{}/{}'.format(split, output_dataset_name)

        with open(file_list_path, 'r') as f:
            file_list = [os.path.join(cfg.path_prefix, line.strip()) for line in f]
        stats = calc_norm_stats(cfg,file_list)
        # Load pretrained weights.
        model = AudioNTT2020Feature(d=cfg.feature_d).cuda()
        weights_path = f'pretrained_weights/AudioNTT2020-BYOLA-64x96d{cfg.feature_d}.pth'
        model.load_weight(weights_path, device)

        # Preprocessor and normalizer.
        to_melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        normalizer = PrecomputedNorm(stats)

        # Optionally fit PCA on all time-step features
        pca = None
        if pca_dim is not None:
            print(f"Fitting PCA to reduce features to {pca_dim} dimensions...")
            feature_samples = []
            for file in tqdm.tqdm(file_list[:pca_fit_samples], desc="Collecting features for PCA"):
                wav, sr = torchaudio.load(file.strip())
                if sr != cfg.sample_rate:
                    resampler = T.Resample(sr, cfg.sample_rate, dtype=wav.dtype)
                    wav = resampler(wav)
                lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())
                with torch.no_grad():
                    features = model(lms.unsqueeze(0).cuda())
                features = features.squeeze(0).cpu().numpy()  # shape [T, D] or [D]
                # Collect all time-step features if sequence
                if features.ndim == 2:
                    feature_samples.append(features)
                else:
                    feature_samples.append(features[np.newaxis, :])
            feature_samples = np.concatenate(feature_samples, axis=0)  # shape [N, D]
            pca = PCA(n_components=pca_dim)
            pca.fit(feature_samples)
            print("PCA fitted.")

        print("Calculating features:")
        for file in tqdm.tqdm(file_list):
        # Load your audio file.
            wav, sr = torchaudio.load(file.strip()) # a sample from SPCV2 for now
            if sr != cfg.sample_rate:
                resampler = T.Resample(sr, cfg.sample_rate, dtype=wav.dtype)
                wav = resampler(wav)
            # assert sr == cfg.sample_rate, "Let's convert the audio sampling rate in advance, or do it here online."

            # Convert to a log-mel spectrogram, then normalize.
            lms = normalizer((to_melspec(wav) + torch.finfo(torch.float).eps).log())

            # Now, convert the audio to the representation.
            with torch.no_grad():
                features = model(lms.unsqueeze(0).cuda())
            features= features.squeeze(0).cpu().numpy()
            # Apply PCA to each time step if needed
            if pca is not None:
                if features.ndim == 2:
                    features = pca.transform(features)  # shape [T, pca_dim]
                else:
                    features = pca.transform(features.reshape(1, -1)).squeeze(0)
            # save output
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)
            base_name = os.path.basename(file.strip()).replace('wav','npy')
            save_name = os.path.join(output_file_path,base_name)
            np.save(save_name,features)
            # print("save features for {}".format(base_name))
            

if __name__ == '__main__':
    run(cfg_path='config.yaml', splits=['dev'])