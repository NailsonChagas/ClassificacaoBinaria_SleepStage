from SleepStageDetection.data.stage import Stages
import os

class Subject:
    def __init__(self, subject_code:int, sample_lenght:int=100, fs:float=100) -> None:
        """
        Armazena os dados do paciente.

        -1 = Sleep stage ?

        0 = Sleep stage W 

        1 = Sleep stage 1 

        2 = Sleep stage 2 

        3 = Sleep stage 3 

        4 = Sleep stage 4 

        5 = Sleep stage R

        Args:
            subject_code (int): Subject code (mne database).
            fs (float, optional): Sampling Frequency. Defaults to 100.
        """        
        self.subject_code = subject_code
        self.sample_lenght = sample_lenght
        self.fs = fs

        self.first_night = Stages(1, self.subject_code, self.sample_lenght, self.fs)
        self.second_night =  Stages(2, self.subject_code, self.sample_lenght, self.fs)


    def get_info(self):
        print(f"Subject code: {self.subject_code}. Sampling Frequency:{self.fs}. Sample lenght: {self.sample_lenght}")
        print("----------------------------------------")
        print("First night:")
        print(self.first_night.get_info())
        print("----------------------------------------")
        print("Second night:")
        print(self.second_night.get_info())
    
    def get_first_night(self):
        return self.first_night
    
    def get_second_night(self):
        return self.second_night
