from SleepStageDetection.feature_extraction.features import Features
import pandas as pd
import time

class Stages: 
    def __init__(self, night:int, subject_code:int, sample_lenght:int=100, fs:float=100) -> None: 
        """
        Armazena os estágios do sono.

        0 = Sleep stage W 

        1 = Sleep stage 1 

        2 = Sleep stage 2 

        3 = Sleep stage 3 

        4 = Sleep stage 4 

        5 = Sleep stage R

        Args:
            night (int): Noite da leitura.
            subject_code (int): Subject code (mne database).
            sample_lenght (int): tamanho desejado dos intervalos a serem analisados. Defaults to 100.
            fs (float, optional): Sampling Frequency. Defaults to 100.
        """     
        self.night = night
        self.code = subject_code 
        self.sl = sample_lenght
        self.fs = fs   
        self.Awake = []
        self.Stage1 = []
        self.Stage2 = []
        self.Stage3 = []
        self.Stage4 = []
        self.Rem_Stage = []
        self.Undefined = 0

    def _extract_sleep_stages(self):
        psg = pd.read_csv("./Database/N" + str(self.night) + "/PSG/" + str(self.code) + "N" + str(self.night) + ".csv")
        hyp = pd.read_csv("./Database/N" + str(self.night) + "/HYP/" + str(self.code) + "N" + str(self.night) + ".csv")
        psg.drop(['EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal', 'Event marker'], axis=1, inplace = True)

        inicio = 0
        for idx, _ in hyp.iterrows():
            final = inicio + hyp['duration'].at[idx] * self.fs #1 leitura a cada 10ms, 100 leituras a cada segundo
            aux = psg.loc[inicio:final].copy() 
            inicio = final

            #percorrer aux a cada "sample_lenght" leituras e jogar dentro do array em que pertence(cada elemento do array sera um df de "sample_lenght" linhas)
            for i in range(0, len(aux.index), self.sl): 
                aux2 = aux.iloc[i:(i+self.sl)].copy()
            
                if hyp['description'].at[idx] == 'Sleep stage W':
                    aux2['description'] = 0
                    self.Awake.append(aux2)
                elif hyp['description'].at[idx] == 'Sleep stage 1':
                    aux2['description'] = 1
                    self.Stage1.append(aux2)
                elif hyp['description'].at[idx] == 'Sleep stage 2':
                    aux2['description'] = 2
                    self.Stage2.append(aux2)    
                elif hyp['description'].at[idx] == 'Sleep stage 3':
                    aux2['description'] = 3
                    self.Stage3.append(aux2)
                elif hyp['description'].at[idx] == 'Sleep stage 4':
                    aux2['description'] = 4
                    self.Stage4.append(aux2)
                elif hyp['description'].at[idx] == 'Sleep stage R':
                    aux2['description'] = 5
                    self.Rem_Stage.append(aux2)
                else:
                    self.Undefined += 1

    def get_info(self):
        print(f"Awake stage (0): {len(self.Awake)}")
        print(f"Stage 1 (1): {len(self.Stage1)}")
        print(f"Stage 2 (2): {len(self.Stage2)}")
        print(f"Stage 3 (3): {len(self.Stage3)}")
        print(f"Stage 4 (4): {len(self.Stage4)}")
        print(f"Rem stage (5): {len(self.Rem_Stage)}")
        print(f"Undefined: {self.Undefined}")
    def get_Awake(self) -> list: #Funciona
        """Retorna estágios Awake

        Returns:
            list: Lista de DataFrames dos estágios Awake
        """           
        return self.Awake
    def get_Stage1(self) -> list: #Funciona
        """Retorna estágios Stage1

        Returns:
            list: Lista de DataFrames dos estágios Stage1
        """           
        return self.Stage1
    def get_Stage2(self) -> list: #Funciona
        """Retorna estágios Stage2

        Returns:
            list: Lista de DataFrames dos estágios Stage2
        """           
        return self.Stage2
    def get_Stage3(self) -> list: #Funciona
        """Retorna estágios Stage3

        Returns:
            list: Lista de DataFrames dos estágios Stage3
        """           
        return self.Stage3
    def get_Stage4(self) -> list: #Funciona
        """Retorna estágios Stage4

        Returns:
            list: Lista de DataFrames dos estágios Stage4
        """           
        return self.Stage4
    def get_Rem_Stage(self) -> list: #Funciona
        """Retorna estágios Rem_Stage

        Returns:
            list: Lista de DataFrames dos estágios Rem_Stage
        """           
        return self.Rem_Stage

    def __awake_features(self):
        features_list = []
        error_count = 0
        cont = 0
        for i in self.get_Awake():
            print(cont)
            cont += 1
            try:
                aux1 = Features(i['EEG Fpz-Cz'], self.fs, 'EEG Fpz-Cz', self.sl)
                aux2 = Features(i['EEG Pz-Oz'], self.fs, 'EEG Pz-Oz', self.sl)
                dictionary = {**aux1.extract_features(), **aux2.extract_features()}
                dictionary.update({'stage':0})
                features_list.append(dictionary)
            except:
                error_count += 1
        #print(error_count)
        return features_list

    def __stage1_features(self):
        features_list = []
        error_count = 0
        cont = 0
        for i in self.get_Stage1():
            print(cont)
            cont += 1
            try:
                aux1 = Features(i['EEG Fpz-Cz'], self.fs, 'EEG Fpz-Cz', self.sl)
                aux2 = Features(i['EEG Pz-Oz'], self.fs, 'EEG Pz-Oz', self.sl)
                dictionary = {**aux1.extract_features(), **aux2.extract_features()}
                dictionary.update({'stage':1})
                features_list.append(dictionary)
            except:
                error_count += 1
        #print(error_count)
        return features_list
    
    def __stage2_features(self):
        features_list = []
        error_count = 0
        cont = 0
        for i in self.get_Stage2():
            print(cont)
            cont += 1
            try:
                aux1 = Features(i['EEG Fpz-Cz'], self.fs, 'EEG Fpz-Cz', self.sl)
                aux2 = Features(i['EEG Pz-Oz'], self.fs, 'EEG Pz-Oz', self.sl)
                dictionary = {**aux1.extract_features(), **aux2.extract_features()}
                dictionary.update({'stage':2})
                features_list.append(dictionary)
            except:
                error_count += 1
        #print(error_count)
        return features_list
    
    def __stage3_features(self):
        features_list = []
        error_count = 0
        cont = 0
        for i in self.get_Stage3():
            print(cont)
            cont += 1
            try:
                aux1 = Features(i['EEG Fpz-Cz'], self.fs, 'EEG Fpz-Cz', self.sl)
                aux2 = Features(i['EEG Pz-Oz'], self.fs, 'EEG Pz-Oz', self.sl)
                dictionary = {**aux1.extract_features(), **aux2.extract_features()}
                dictionary.update({'stage':3})
                features_list.append(dictionary)
            except:
                error_count += 1
        #print(error_count)
        return features_list

    def __stage4_features(self):
        features_list = []
        error_count = 0
        cont = 0
        for i in self.get_Stage4():
            print(cont)
            cont += 1
            try:
                aux1 = Features(i['EEG Fpz-Cz'], self.fs, 'EEG Fpz-Cz', self.sl)
                aux2 = Features(i['EEG Pz-Oz'], self.fs, 'EEG Pz-Oz', self.sl)
                dictionary = {**aux1.extract_features(), **aux2.extract_features()}
                dictionary.update({'stage':4})
                features_list.append(dictionary)
            except:
                error_count += 1
        #print(error_count)
        return features_list
    
    def __rem_features(self):
        features_list = []
        error_count = 0
        cont = 0
        for i in self.get_Rem_Stage():
            print(cont)
            cont += 1
            try:
                aux1 = Features(i['EEG Fpz-Cz'], self.fs, 'EEG Fpz-Cz', self.sl)
                aux2 = Features(i['EEG Pz-Oz'], self.fs, 'EEG Pz-Oz', self.sl)
                dictionary = {**aux1.extract_features(), **aux2.extract_features()}
                dictionary.update({'stage':5})
                features_list.append(dictionary)
            except:
                error_count += 1
        #print(error_count)
        return features_list
    
    def extract_features(self): #funcionando
        #print("entrou")#remover
        inicio = time.time()
        self._extract_sleep_stages()
        features_list = self.__awake_features() + self.__stage1_features() + self.__stage2_features() + self.__stage3_features() + self.__stage4_features() + self.__rem_features()
        data = pd.DataFrame(features_list)
        path = f"./Features/{str(self.code)}N{str(self.night)}_{str(self.sl)}_{str(self.fs)}.csv"
        try:
            data.to_csv(path, index=False)
        except Exception as e:
            print(e)
        fim = time.time()
        tempo_exec = fim - inicio
        return tempo_exec

    def remove_NaN_colums(self, thresh_hold:int=3000):
        #Essa função só foi feita para que eu n tenha q remover todas as colunas do arquivo original
        novo = pd.read_csv(f"./Features/{str(self.code)}N{str(self.night)}_{str(self.sl)}_{str(self.fs)}.csv")
        pd.options.mode.use_inf_as_na = True
        novo.dropna(axis=1, how='any', thresh=(len(novo.index) - thresh_hold), inplace=True)
        novo.to_csv(f"./Features/{str(self.code)}N{str(self.night)}_{str(self.sl)}_{str(self.fs)}_noNaN_{thresh_hold}.csv", index=False)

    def get_features(self):
        return pd.read_csv(f"./Features/{str(self.code)}N{str(self.night)}_{str(self.sl)}_{str(self.fs)}.csv")
        
    def get_noNaN_features(self, thresh_hold:int=3000):
        return pd.read_csv(f"./Features/{str(self.code)}N{str(self.night)}_{str(self.sl)}_{str(self.fs)}_noNaN_{thresh_hold}.csv")

if __name__ == "__main__":
    pass
    #teste = Stages(2,0)
    #tempo, tabela = teste.save_stage_features()
    #print(tabela)
    #tabela.info()
    #print(tempo)
