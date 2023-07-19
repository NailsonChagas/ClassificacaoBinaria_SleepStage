# Detecção de estagios do sono utilizando aprendizado de maquina

Projeto de iniciação científica<br>
Campus: UTFPR - Pato Branco<br>
Orientador: Jefferson T. Oliva<br>
Curso: Engenharia de Computação.

**Obs:** Esse repositório é a atualização de outro repositório já existente em minha conta (SleepStageDetection).

## Feito
<ul>
    <li>Download da base de dados Sleep-EDF através do pacote MNE (https://mne.tools/dev/install/index.html).</li>
    <li>Separando os dados por fases do sono.</li>
    <li>Remoção de leituras invalidas.</li>
    <li>Utilizando a biblioteca sfe (ainda não publicada, autoria: Jefferson T. Oliva) para a extração de características.</li>
    <li>Remoção de todas colunas do DataFrame de caracteristicas que possum algum valor NaN</li>
    <li>Armazena as caracteristicas extraidas na pasta Features (<b>Obs:</b> nomes seguem esse formato "./Features/{subject_code}N{night}_{sample_length}_{frequency_sample}.csv").</li>
    <li>Testando classificadores no tipo neighbors e ensemble</li>
    <li>Pontuação dos classificadores salvos na pasta "./Scores"</li>
    <li>Implementação do balanceamento dos dataframes</li>
    <li>Implementar classificação binária</li>
    <li>Comparar com multiclasse</li>
    <li>Agrupar amostras por idade</li>
    <li>Comparar com as classificações feitas com uma unica amostra</li>
</ul>

## A Fazer
<ul>
    
    
</ul>

<hr>

### Erros:
<ul>
    <li>line length: all nan</li>
    <li>nonlinear energy: all nan</li>
    <li>Hurst expoent: all nan</li>
    <li>Shannon entropy: all nan</li>
    <li>Renvi entropy: all nan</li>
</ul>

### Erros resolvidos
<ul>
    <li>
        Durante os testes dos classificadores tipo neighbors e ensemble algumas colunas do dataframe apresentaram erros:
        <ul>
            <li>'sample_entropy_PS_Gama_EEG Fpz-Cz'</li>
            <li>'sample_entropy_PS_Entire_EEG Pz-Oz'</li>
            <li>'sample_entropy_PS_Entire_EEG Fpz-Cz'</li>
            <li>'sample_entropy_TS_Entire_EEG Fpz-Cz'</li>
            <li>'sample_entropy_PS_Gama_EEG Pz-Oz'</li>
            <li>'sample_entropy_TS_Entire_EEG Pz-Oz'</li>
        </ul>
        Problema encontrado: por padrão o pandas não considera inf e -inf como NaN<br>
        Solução: pandas.options.mode.use_inf_as_na = True
    </li>
</ul>

<hr>

## Fases do sono

Fase      | Representante 
--------- | ---------
Awake     | 0
Stage 1   | 1
Stage 2   | 2
Stage 3   | 3
Stage 4   | 4
Rem       | 5
Undefined | -1

---

## Características Extraidas

#### Série Temporal

Banda     | Qtd. de características
--------- | ------
Inteiro   | 27
**Total** | **27**

#### Espectro de potência

Banda     | Qtd. de características
--------- | ------
Delta     | 29
Theta     | 29
Alpha     | 29
Beta      | 29
Gamma     | 29
Inteiro   | 29
**Total** | **174**

#### Espectrograma

Banda     | Qtd. de características
--------- | ------
Delta     | 16
Theta     | 16
Alpha     | 16
Beta      | 16
Gamma     | 16
Inteiro   | 16
**Total** | **96**

Obs: No total 297 características
