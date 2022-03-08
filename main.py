from SleepStageDetection.data.subject import Subject

#teste = Subject(0)
#teste.get_info()
#tempo = teste.get_first_night().extract_features()
#print(tempo)
#teste.get_first_night().remove_NaN_colums(2000)
#print("--------------------------------------")
#teste.get_second_night().extract_features()
#teste.get_second_night().remove_NaN_colums(2000)

#print(len(teste.get_first_night().get_noNaN_features(2000).columns))
#data = teste.get_first_night().get_noNaN_features(2000)
#count = get_class_count(data, 'stage')


def extract_features_in_range(ini:int = 0, end:int = 83):
    errors = []
    for i in range(ini, end):
        try:
            subject = Subject(i)
            
            subject.get_first_night().extract_features()
            subject.get_first_night().remove_NaN_colums()

            subject.get_second_night().extract_features()
            subject.get_second_night().remove_NaN_colums()

            del subject
        except Exception as e:
            errors.append(e)
    print(errors)

def remove_nan(ini:int = 0, end:int = 83):
    errors = []
    for i in range(ini, end):
        try:
            subject = Subject(i)
            
            subject.get_first_night().remove_NaN_colums()
            subject.get_second_night().remove_NaN_colums()

            del subject
        except Exception as e:
            errors.append(e)
    print(errors)
    
extract_features_in_range(6, 83)