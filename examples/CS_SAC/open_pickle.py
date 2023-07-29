import pandas as pd

pickleFile = open("/disk/nfs/ostrom/s2324992/data/LJSpeech-1.1/word2speechreps/ljspeech_test_km100_word2speechreps.pickle","rb")

obj = pd.read_pickle(pickleFile)
# print (obj)
with open('/home/s2324992/fairseq_new/examples/CS_SAC/pickle_eng100.txt', 'w') as output_file:
    output_file.write("\n".join([f"{i}\t0" for i in obj]))
    # with open(outfile, 'w') as f:
    # f.write("\n".join([f"{wav_path}\t0" for wav_path in files]))