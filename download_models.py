import os
import gdown
import shutil

files_ids = [
'1xc0dUD4mjCLOdbQSAhwGVYUmBsqqrcxF',
'1-TP6nM4MynjmtXNMoHsA8mz8c5gbyaoB',
'1lm-Vsw_PhAUh8w-K6yY-rxgDJShbb6E7',
'1s39TRtZt7vqX9btIUgHscdvwEoF_V4cs',
'1KPXcRzl2w_TdIuFUV6c7fbM5QyFUorzU',
'118bDpmrXr8D7to7VYy4BYXZ_HLIEGvy0',
'1eHafsQIsc2fzO-lArZ-mVhMxjUaWdETm',
'1BrKvozBFQRPCVxABJt9Qd9yUP_BQ-lTF',
'1-L_7bI_keaoNPjeXoJhlywi5dt19lSUE',
'1T3QfhHIY1T0P228BVxqMBEguK-391UXc'
]

better_id = '1_guU3Pxe8tTmEppelUU5Wb4XI1dgKRdr'

if __name__ == "__main__":
    main_path = os.path.dirname(os.path.realpath(__file__))+'/'
    n_experiments = 10
    for i in range(n_experiments):
        file_id = files_ids[i]
        experiment_path = main_path+'experiments/experiment'+str(i+1)+'/resnet34.pth'
        os.remove(experiment_path)
        #gdown.download('https://drive.google.com/uc?id='+file_id, experiment_path, quiet=False)
    
    #gdown.download('https://drive.google.com/uc?id='+better_id, main_path+'/BetterResults/results.zip', quiet=False)
    #shutil.unpack_archive( main_path+'/BetterResults/results.zip', main_path+'/BetterResults/')
