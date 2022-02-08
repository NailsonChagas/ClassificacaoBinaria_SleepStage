from mne.datasets.sleep_physionet.age import fetch_data
import shutil
import mne
import os

def download_database(database_output:str, PSG_output:str, HYP_output:str, night:int) -> None: #Converter o HYP depois
    """Dowload do banco de dados.

    Args:
        database_output (str): caminho de saida
        PSG_output (str): caminho de saída do PSG
        HYP_output (str): caminho de saída do HYP
        night (int): noite de leitura
    """    
    b = 0
    for i in range(0,83):
        if i not in [13, 36, 39, 52, 68, 69, 78, 79]: #13, 36 e 52: faltam dados da noite 2 / following subjects are not available: 39, 68, 69, 78 and 79
            [i] = fetch_data(subjects=[i],recording=[night], path=database_output, on_missing='ignore')
            raw_edf = mne.io.read_raw_edf(i[0])
            annot = mne.read_annotations(i[1])
            
            edf_annot = raw_edf.set_annotations(annot, emit_warning=False)
            edf_final_path = PSG_output + "/" + str(b) + "N" + str(night) + ".csv"
            hyp_final_path = HYP_output + "/" + str(b) + "N" + str(night) + ".csv"
            annot.save(hyp_final_path)
            data = edf_annot.to_data_frame()
            data.to_csv(edf_final_path, index=False)
        b += 1
    print("Completo")

def dowload_subject(database_output:str, subject:int, night:int) -> None:
    """Download de um unico paciente.

    Args:
        database_output (str): caminho de saida
        subject (int): códico do paciente
        night (int): noite de leitura
    """    
    [s] = fetch_data(subjects=[subject], recording=[night], path=database_output, on_missing='ignore')

if __name__ == '__main__':
    temp = "./Database/temp"
    output_PSG_1 = "./Database/N1/PSG"
    output_HYP_1 = "./Database/N1/HYP"
    output_PSG_2 = "./Database/N2/PSG"
    output_HYP_2 = "./Database/N2/HYP"

    if os.path.isdir('./Database') != True:
        os.makedirs(output_PSG_1, exist_ok=True)
        os.makedirs(output_HYP_1, exist_ok=True)
        os.makedirs(output_PSG_2, exist_ok=True)
        os.makedirs(output_HYP_2, exist_ok=True)
    
    try:
        os.mkdir(temp)
        download_database(temp, output_PSG_1, output_HYP_1, 1)
        shutil.rmtree(temp)
        os.mkdir(temp)
        download_database(temp, output_PSG_2, output_HYP_2, 2)
        shutil.rmtree(temp)
    except Exception as e:
        print(f"Erro: {e}")