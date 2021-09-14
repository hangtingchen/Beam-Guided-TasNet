import json
import sys

basepath='local_decodespwsj2/'

if __name__=="__main__":
    data_dir=sys.argv[1]
    wav_dir=sys.argv[2]
    with open(basepath+'rir_info.json','r') as f:
        rir_info = json.load(f)['INFO']
    text = dict();text[1]=dict();text[2]=dict()
    with open(basepath+'text_spk1') as f:
        for line in f:
            k,s = line.strip().split(" ",maxsplit=1)
            text[1][k[8:]]=s
    with open(basepath+'text_spk2') as f:
        for line in f:
            k,s = line.strip().split(" ",maxsplit=1)
            text[2][k[8:]]=s
    if(data_dir.find('cv')>=0):
        dataset='cv'
        rir_info = rir_info[20000:25000]
        offset = 20000
        with open(basepath+'wsj0-2mix_cv.flist','r') as f:
            flist = f.read().strip().split('\n')
    elif(data_dir.find('tt')>=0):
        dataset='tt'
        offset = 25000
        rir_info = rir_info[25000:28000]
        with open(basepath+'wsj0-2mix_tt.flist','r') as f:
            flist = f.read().strip().split('\n')
    elif(data_dir.find('tr')>=0):
        dataset='tr'
        offset = 0
        rir_info = rir_info[0:25000]
        with open(basepath+'wsj0-2mix_tr.flist','r') as f:
            flist = f.read().strip().split('\n')
    else:
        print(data_dir)
        raise NotImplementedError
    with open(data_dir+'/wav.scp','w') as fw, open(data_dir+'/utt2spk','w') as ft,open(data_dir+'/text','w') as fu:
        for i,k in enumerate(flist):
            spk1,_,spk2,_=k[:-4].split("_")            
            fw.write("{}_{}_{}_{} {}/s1/{}\n".format(spk1[0:3],i+offset,spk1,spk2,wav_dir,k))
            fw.write("{}_{}_{}_{} {}/s2/{}\n".format(spk2[0:3],i+offset,spk1,spk2,wav_dir,k))
            ft.write("{}_{}_{}_{} {}\n".format(spk1[0:3],i+offset,spk1,spk2,spk1[0:3]))
            ft.write("{}_{}_{}_{} {}\n".format(spk2[0:3],i+offset,spk1,spk2,spk2[0:3]))
            fu.write("{}_{}_{}_{} {}\n".format(spk1[0:3],i+offset,spk1,spk2,text[1][k[:-4]]))
            fu.write("{}_{}_{}_{} {}\n".format(spk2[0:3],i+offset,spk1,spk2,text[2][k[:-4]]))
