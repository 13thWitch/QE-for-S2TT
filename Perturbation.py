import resampy
import numpy as np
from scipy.fft import fft, ifft
from pydub.silence import detect_leading_silence
from pydub import AudioSegment

class Perturbator:
    """
    This class is an audio perturbation resource. It offers whole-audio, aas well as segment-level perturbations, if given a transcription.
    Perturbations include resampling, warping, random noise addition, and frequency filtering.
    @param config: dictionary of perturbation strategies and their parameters. Example: 
    {
        "resampling": {
            "target_sample_rates": [8000, 16000, 32000]
        },
        "speed_warp": {
            "speeds": [0.5, 1, 2]
        },
        "random_noise": {
            "std_ns": [0.01, 0.05, 0.1]
        },
        "frequency_filtering": {
            "pass_cutoffs": [(100, 200), (300, 400)],
            "stop_cutoffs": [(500, 600), (700, 800)]
    }
    """
    def __init__(self, config) -> None:
        # TODO: reconsider transcription use: maybe do two big blocks of pertutbation, keep instructions the same, but add token-focus wrapper with corresponding audio split to one block
        self.instruction = dict(
            {
                "resampling": {
                    "method": lambda audio, sr: self.resample(audio, sample_rate=sr),
                    "use_transcription": False
                }, 
                "frequency_filtering" : {
                    "method": lambda audio, sr: self.frequency_filtering(audio, sample_rate=sr),
                }, 
                "random_noise": {
                    "method": lambda audio, sr: self.random_noise(audio, sample_rate=sr)
                },
                "speed_warp": {
                    "method": lambda audio, sr: self.speed_warp(audio, sample_rate=sr) 
                }
            }
        )
        for perturbation_strat, parameters in config.items():
            for param, value in parameters.items():
                self.instruction[perturbation_strat][param] = value
        
        self.instruction = {k: v for k, v in self.instruction.items() if k in config.keys()}
        
    
    def resample(self, audio, sample_rate):
        """
        Resamples the audio to instructed sample rates.
        @param audio: numpy ndarray of audio
        @param sample_rate: original sample rate integer
        @return dict containing pairs of target sample rate and corresponding resampled audio
        """
        result = dict()
        for sr in self.instruction["resampling"]["target_sample_rates"]:
            result[str(sr)] = audio if str(sr) == str(sample_rate) else resampy.resample(audio, sample_rate, sr)
        return result
    
    def speed_warp(self, audio, sample_rate):
        """
        Resamples the audio to its original sample rate using modified source sample rate to mimic speed changes.
        Speed parameter value meaning: 
        - value < 1 slows audio down to value of original speed
        - value == 1 changes nothing
        - value > 1 speeds audio up to value of original audio.
        Example: a value of 0.5 would make the audio sound half as fast and be twice as long.
        @param audio: numpy ndarray of original audio
        @param sample_rate: integer of original sample rate
        @return dict: pairs of speed and corresponding perturbed audio
        """
        result = dict()
        for speed in self.instruction["speed_warp"]["speeds"]:
            result[str(speed)] = audio if speed == 1 else resampy.resample(audio, sample_rate*speed, sample_rate)
        return result
    
    def random_noise(self, audio, sample_rate):
        """
        Adds random noise to given audio based on instructed standard deviation.
        @param audio as numpy ndarray
        @param sample rate of audio as int
        @returns dict containing standard deviatons and the corresponding noisy audio
        """
        result = dict()
        # TODO: add pink, brown, white etc.
        for std_n in self.instruction["random_noise"]["std_ns"]:
            noise = np.random.normal(0, std_n, audio.shape[0])
            noisy = audio + noise
            result[str(std_n)] = noisy
        return result

    def frequency_filtering(self, audio, sample_rate):
        """
        Filters given audio based on instructed frequency ranges using band-pass or band-stop filters, as instructed.
        @param audio: numpy ndarray of audio
        @param sample_rate: int given audio's sample rate
        @returns dict containing pairs of frequency ranges, the filter type (pass or stop) and corresponding filtered audio
        """
        result = dict()
        # Fourier Transform to convert the audio signal to the frequency domain
        audio_fft = fft(audio)
        # Generate frequency bins
        num_samples = len(audio_fft)
        freqs = np.fft.fftfreq(num_samples, 1/sample_rate)

        if "pass_cutoffs" in self.instruction["frequency_filtering"].keys():
            for (lower, upper) in self.instruction["frequency_filtering"]["pass_cutoffs"]:
                audio_fft_pass = audio_fft.copy()
                # Zero out components not within the band-pass range
                for i in range(num_samples//2):
                    if freqs[i] < lower or freqs[i+1] > upper:
                        audio_fft_pass[i] = 0
                        audio_fft_pass[-(i-1)] = 0

                # Inverse Fourier Transform to convert back to the time domain
                result[f"pass{str((lower, upper))}"] = ifft(audio_fft_pass).real  # Take the real part
        
        if not "stop_cutoffs" in self.instruction["frequency_filtering"].keys():
            # no stop filtering requested, thus we are done
            return result
        
        for (lower, upper) in self.instruction["frequency_filtering"]["stop_cutoffs"]:
            audio_fft_stop = audio_fft.copy()
            # Zero out components within the band-stop range
            for i in range(num_samples//2):
                if freqs[i] >= lower and freqs[i+1] <= upper:
                    audio_fft_stop[i] = 0
                    audio_fft_stop[-(i-1)] = 0
            result[f"stop{str((lower, upper))}"] = ifft(audio_fft_stop).real
            
        return result

    def get_perturbations(self, audio, sample_rate, transcription="", combined=False):
        """
        Produce perturbations of given audio based on instance-wide instruction set. 
        @param audio: numpy ndarray of audio
        @param sample_rate: int given audio's sample rate
        @param transcription[optional]: string audio's transcribed speech
        @param combined[optional]: boolean indicating whether to use transcription as well as unsegmented perturbations
        @returns a dictionary of perturbation types and respective resulting audio. If transcription is used, depending on value of combined parameter, the dict contains only or additionally includes segmented perturbations.
        """
        audio = trim_silence(audio, sample_rate)
        if transcription:
            segmented_perturbations = self.use_transcription(audio, sample_rate, transcription)
            if not combined:
                return segmented_perturbations
        perturbations = dict()
        for perturbation_strategy, specification in self.instruction.items():
            perturbations[perturbation_strategy] = specification["method"](audio, sample_rate)
        # return perturbations
        if transcription:
            return segmented_perturbations.update(flatten_dict(perturbations))
        return flatten_dict(perturbations)
    
    def use_transcription(self, audio, sample_rate, transcription):
        """
        Segment audio based on transcription and apply perturbations to each segment.
        @param audio: numpy ndarray of audio
        @param sample_rate: int given audio's sample rate
        @param transcription: string audio's transcribed speech
        @returns a dictionary of perturbation types and respective resulting audio. Keys are formatted as "seg-{segment_number}_{perturbation_strategy}"
        """
        # TODO: allocate segment size based on word length/phoneme count.
        # Use nltk or similar to tokenize transcription? 
        # TODO: add character-language support list(u"这是一个句子")
        perturbations = dict()
        num_words = len(transcription.split())
        segment_length = len(audio) // num_words
        for i in range(num_words):
            segment = audio[i*segment_length:(i+1)*segment_length]
            for specification in self.instruction.values():
                # apply perturbation strategy to segment, recieve dict of different variants
                perturbed_segments = specification["method"](segment, sample_rate)
                for perturbation, perturbed_segment in perturbed_segments.items():
                    # for each perturb. param. variant, reattach to audio and store in dict
                    new_audio = np.concatenate([audio[:i*segment_length], perturbed_segment, audio[(i+1)*segment_length:]])
                    perturbations[f"seg-{i+1}_{perturbation}"] = new_audio
        
        return perturbations
    
def flatten_dict(dictionary):
        """
        Flatten a dictionary of dictionaries to a single dictionary by concatenating keys along the path.
        @param dictionary: dictionary of dictionaries
        @returns a single dictionary
        """
        return {f"{super_key}-{k}": v for super_key, subdict in dictionary.items() for k, v in subdict.items()}

def trim_silence(audio, sr):
    """
    Trim silence from the beginning and end of audio.
    @param audio: numpy ndarray of audio
    @returns trimmed audio
    # TODO: potential time complexity improvement here. convert to int16 with 32767, then back to float32.
    """
    array = np.int32(audio * 2147483647)
    audio_segment = AudioSegment(array.tobytes(), frame_rate=sr, sample_width=array.dtype.itemsize, channels=1)
    trim_leading_silence = lambda x: x[detect_leading_silence(x) :]
    trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
    strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))
    return np.float32(np.array(strip_silence(audio_segment).get_array_of_samples()) / 2147483647)
