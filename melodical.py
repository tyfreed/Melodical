import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display


class Melodical(object):
    def __init__(self, mp3_path) -> None:
        # Calculated values
        # Start
        self.key = None
        self.key_correlation = None
        # End

        # Constants
        # Start
        self.pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.major = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        self.minor = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        self.keys = ['C major', 'C# major', 'D major', 'D# major', 'E major', 'F major', 'F# major', 'G major', 'G# major',
                  'A major', 'A# major', 'B major', 'C minor', 'C# minor', 'D minor', 'D# minor', 'E minor', 'F minor',
                  'F# minor', 'G minor', 'G# minor', 'A minor', 'A# minor', 'B minor']
        # End

        # Calculated values basaed on user input.
        # Start
        self.mp3_path = mp3_path
        self.y, self.sample_rate = librosa.load(self.mp3_path)
        self.y_harmonic = librosa.effects.harmonic(self.y)
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_harmonic, sr=self.sample_rate,
                                                      bins_per_octave=24)  # Investigate octave error on BPM's

        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        # End

        self._calculate_key()

        pass

    def _calculate_key(self) -> None:
        major_correlation = []
        minor_correlation = []
        keyfreqs = {self.pitches[i]: self.chroma_vals[i] for i in range(12)}

        for i in range(12):
            key_test = [keyfreqs.get(self.pitches[(i + m) % 12]) for m in range(12)]
            major_correlation.append(round(np.corrcoef(self.major, key_test)[1, 0], 3))
            minor_correlation.append(round(np.corrcoef(self.minor, key_test)[1, 0], 3))

        key_correlations = major_correlation + minor_correlation

        max_correlation = max(key_correlations)
        max_index = key_correlations.index(max_correlation)

        self.key = self.keys[max_index]
        self.key_correlation = max_correlation

    def get_key(self) -> str:
        return self.key

