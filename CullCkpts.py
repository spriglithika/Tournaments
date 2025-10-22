import os

for path, directories, files in os.walk('ckpts'):
    if len(files) > 0:
        best_base = 0
        best_mid = 0
        best_tourn = 0
        for file in files:
            if 'NA' in file:
                continue
            terms = file.split('_')
            epoch_ = terms[2][5:-4]
            if epoch_ == '':
                continue
            epoch = int(terms[2][5:-4])
            if terms[0] == 'base' and epoch > best_base:
                best_base = epoch
            if terms[0] == 'mid' and epoch > best_mid:
                best_mid = epoch
            if terms[0] == 'tournament' and epoch > best_tourn:
                best_tourn = epoch
        print(best_base, best_mid, best_tourn)
        for file in files: 
            if file not in [f'base_best_epoch{best_base}.pth',f'mid_best_epoch{best_mid}.pth',f'tournament_best_epoch{best_tourn}.pth']:
                # continue
                os.remove(path + '/' + file)
            # else:
                # print(file)