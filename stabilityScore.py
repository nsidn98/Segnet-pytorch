import numpy as np
import os
from tqdm import tqdm
if not os.path.exists('./Train'):
    os.makedirs('./Train')
if not os.path.exists('./Train_annot'):
    os.makedirs('./Train_annot')

cell_len = 80
cell_wid = 45
cell_ht  = 45
max_ldc_x  = 1
max_ldc_y =1


def getStabilityScore(i, j , ldc, dimn, currldc_x, currldc_y):
    level = ldc[i,j]
    h = dimn[2]
    feasible = False
    found_flat = found_not_flat = 0
    if  (j >= cell_len*currldc_x) and (j+dimn[0] <= cell_len*(currldc_x+1)) and\
        (i >= cell_wid*currldc_y) and (i+dimn[1] <= cell_wid*(currldc_y+1)) and\
        level + h <= cell_ht:
        feasible = True
        # --------------------------------------------------- Flat position
        if len(np.unique(ldc[i:i+dimn[1], j:j+dimn[0]])) == 1:
            stab_score = 1
            found_flat = 1
        # ---------------------------------------------------- Non-Flat position
        if not found_flat:
            corners =  [ldc[i,j], ldc[i+dimn[1]-1,j], ldc[i,j+dimn[0]-1], ldc[i+dimn[1]-1, j+dimn[0]-1]]
            if (np.max(corners) == np.min(corners)) and (np.max(corners) == np.max(ldc[i:i+dimn[1],j:j+dimn[0]])):
                stab_score = - np.sum(np.max(corners)-ldc[i:i+dimn[1],j:j+dimn[0]])/(dimn[0]*dimn[1]*cell_ht)
                found_not_flat = 1


    if (found_flat) or (found_not_flat):
        minj = np.max((cell_len*currldc_x,j-1))
        maxj = np.min((cell_len*(currldc_x+1),j+dimn[0]))
        mini = np.max((cell_wid*(currldc_y),i-1))
        maxi = np.min((cell_wid*(currldc_y+1),i+dimn[1]))

        # Border for the upper edge
        if i==currldc_y*cell_wid: 
            upper_border = (cell_ht - 1 + np.ones_like(ldc[mini,j:(j+int(dimn[0]))])).tolist()
        else: 
            upper_border  = ldc[mini,j:(j+int(dimn[0]))].tolist()
        # Stability for the upper edge
        unique_ht = np.unique(upper_border)
        if len(unique_ht) == 1:
            stab_score += 0.5
            if unique_ht[0] == level: stab_score -= 2
            elif unique_ht[0] == cell_ht: stab_score += 1.5
            else:
                sscore = 1.-abs(unique_ht[0]-(level+h))/cell_ht
                if (unique_ht[0]>level): stab_score += 1.5*sscore
                else:                    stab_score += 0.75*sscore
        else:
            stab_score += 0.25*(1.-len(unique_ht)/h)
            stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
            stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
        #border.extend(upper_border)
        del upper_border

        # Border for the left edge
        if j==currldc_x*cell_len:
            left_border = (cell_ht - 1 + np.ones_like(ldc[i:(i+int(dimn[1])),minj])).tolist()
        else: 
            left_border = ldc[i:(i+int(dimn[1])),minj].tolist()
        # Stability for the left edge
        unique_ht = np.unique(left_border)
        if len(unique_ht) == 1:
            stab_score += 0.5
            if unique_ht[0] == level: stab_score -= 2
            elif unique_ht[0] == cell_ht: stab_score += 1.5
            else:
                sscore = 1.-abs(unique_ht[0]-(level+h))/cell_ht
                if (unique_ht[0]>level): stab_score += 1.5*sscore
                else:                    stab_score += 0.75*sscore
        else:
            stab_score += 0.25*(1.-len(unique_ht)/h)
            stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
            stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
        #border.extend(left_border)
        del left_border

        # Border for the lower edge
        if (i+dimn[1] < cell_wid*(currldc_y+1)): lower_border = ldc[maxi,j:(j+int(dimn[0]))].tolist()
        else: lower_border = (cell_ht - 1 + np.ones_like(ldc[maxi-1,j:(j+int(dimn[0]))])).tolist()
        # Stability for the lower edge
        unique_ht = np.unique(lower_border)
        if len(unique_ht) == 1:
            stab_score += 0.5
            if lower_border[0] == level: stab_score -= 2
            elif lower_border[0] == cell_ht: stab_score += 1.5
            else:
                sscore = 1.-abs(unique_ht[0]-(level+h))/cell_ht
                if (unique_ht[0]>level): stab_score += 1.5*sscore
                else:                    stab_score += 0.75*sscore
        else:
            stab_score += 0.25*(1.-len(unique_ht)/h)
            stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
            stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
        #border.extend(lower_border)
        del lower_border

        # Border for the right edge
        if (j+dimn[0] < (currldc_x+1)*cell_len): right_border = ldc[i:(i+int(dimn[1])),maxj].tolist()
        else: 
            right_border = (cell_ht - 1 + np.ones_like(ldc[i:(i+int(dimn[1])),maxj-1])).tolist()
        # Stability for the right edge
        unique_ht = np.unique(right_border)
        if len(unique_ht) == 1:
            stab_score += 0.5
            if right_border[0] == level: 
                stab_score -= 2
            elif right_border[0] == cell_ht: 
                stab_score += 1.5
            else:
                sscore = 1.-abs(unique_ht[0]-(level+h))/cell_ht
                if (unique_ht[0]>level): 
                    stab_score += 1.5*sscore
                else:                    
                    stab_score += 0.75*sscore
        else:
            stab_score += 0.25*(1.-len(unique_ht)/h)
            stab_score += 0.25*(1.-sum(abs(ht-(level+h)) for ht in unique_ht)/(h*len(unique_ht)))
            stab_score += 0.50*sum(ht!=level for ht in unique_ht)/len(unique_ht)
        #border.extend(right_border)
        del right_border
        
        
        # Check the upper edge for continuity
        if i == currldc_y*cell_wid: stab_score += 0.02
        else:
            # In the upper-left corner
            if (j == currldc_x*cell_len) :
                stab_score += 0.01
            # In the upper-right corner
            if ((j+dimn[0]) == (currldc_x+1)*cell_len) :
                stab_score += 0.01
        # Check the lower edge for continuity
        if i+dimn[1] == cell_wid*(currldc_y+1): 
            stab_score += 0.02
        else:
            # In the lower-left corner
            if (j == currldc_x*cell_len) :
                stab_score += 0.01
            # In the lower-right corner
            if ((j+dimn[0]) == (currldc_x+1)*cell_len) :
                stab_score += 0.01
        # Check the left edge for continuity
        if j == currldc_x*cell_len: 
            stab_score += 0.02
        else:
            # In the upper-left corner
            if (i == currldc_y*cell_wid): 
                stab_score += 0.01
            # In the lower-left corner
            if (i+dimn[1] == cell_wid*(currldc_y+1)): 
                stab_score += 0.01
        # Check the right edge for continuity 
        if j+dimn[0] == (currldc_x+1)*cell_len: 
            stab_score += 0.02
        else:
            # In the upper-left corner
            if (i == currldc_y*cell_wid) : 
                stab_score += 0.01
            # In the lower-left corner
            if (i+dimn[1] == cell_wid*(currldc_y+1)): 
                stab_score += 0.01 
            
        stab_score -= currldc_x/max_ldc_x + currldc_y/max_ldc_y
        stab_score -= 0.05*(i/((currldc_y+1)*cell_wid) + j/((currldc_x+1)*cell_len))
        stab_score -= level / cell_ht
        #stab_score += (dimn[0]*dimn[1])/(cell_len*cell_wid)
    else:
        stab_score = -10
        
    return stab_score

def get_final_img(dim,img,normalise=1):
    shape = img.shape
    final_img = np.zeros((shape[0],shape[1],3))
    channel2 = np.zeros((shape[0],shape[1]))
    channel2[:dim[1],:dim[0]] = dim[2]
    channel3 = np.zeros((shape[0],shape[1]))
    channel3[:,shape[1]-dim[0]+1:] = 45
    channel3[shape[0]-dim[1]+1:,:] = 45
    final_img[:,:,0] = img
    final_img[:,:,1] = channel2
    final_img[:,:,2] = channel3
    if normalise:
        return final_img/45
    else:
        return final_img
    
for epoch in tqdm(range(100)):
    episode_num = np.random.randint(0,11,1)[0]
    for step in range(len(os.listdir('States/episode_'+str(episode_num)))):
        ldc = np.load('States/episode_'+str(episode_num)+'/step_'+str(step)+'.npy')[:,:80]
        score = np.zeros(ldc.shape)
        l=np.random.randint(2,40,1)[0]
        b=np.random.randint(2,40,1)[0]
        h=np.random.randint(2,40,1)[0]
        for i in range(score.shape[0]):
            for j in range(score.shape[1]):
                score[i,j] = np.round(getStabilityScore(i, j , ldc=ldc, dimn=[l,b,h], currldc_x=0, currldc_y=0))
        np.save('Train_annot/annot_'+str(epoch)+'_'+str(episode_num)+'_'+str(step)+'.npy',score,allow_pickle=True)
        img = get_final_img([l,b,h],ldc,0)
        np.save('Train/train_'+str(epoch)+'_'+str(episode_num)+'_'+str(step)+'.npy',img,allow_pickle=True)
