import numpy as np
import pandas as pd
import math
from keras.preprocessing.image import  img_to_array
from skimage import transform
from deep_ranking import deep_rank_model
import skimage
import matplotlib.pyplot as plt

# =============================================================================
# 
# =============================================================================
def MyfGetRcImgM1(s, d, tau):

    y = np.transpose(s)
    N = len(y)
    N2 = N - tau * (d - 1)

    xe = []
    for mi in range(d):
        y1 = y[0:N2]
        te = []
        for i in range(len(y1)):
            te.append(y1[i] + tau * mi)
        xe.append(te)
    
    xe = np.transpose(xe) 
    x1 = np.tile(xe, [N2, 1])    
    xe = np.transpose(xe)    
    m1 = np.reshape(xe,(N2*d , 1))
    mm = np.tile(m1, (1 , N2));
    x2 = np.reshape(mm,(d,N2*N2))
    x2 = np.transpose(x2)    
    f1 = x1-x2
    
    f2 = []
    for i in range(d):
        ff = f1[:,i]
        s1 = []
        for j in ff:
            s1.append(math.pow(j,2))
        f2.append(s1)    
    f2 = np.transpose(f2)
    
    S = []
    for row in f2:
        S.append(math.sqrt(sum(row)))
    S = np.reshape(S, (N2, N2))
    
    return np.array(S)



# =============================================================================
#  create series and label
# =============================================================================
def create_series(dataset,seq_length, method):
    X = []
    Y = []
    length = len(dataset)
    
    if method == 'continues':
        for i in range(0, length-seq_length, 1):
             sequence = dataset[i:i + seq_length]
             label = dataset[i + seq_length]
             X.append( sequence)
             Y.append(label)
    else:
        for i in range(0, length, seq_length):
             sequence = dataset[i:i + seq_length-1]
             label = dataset[i + seq_length-1]
             X.append( sequence)
             Y.append(label)
      
    return X,Y



# =============================================================================
#  load data
# =============================================================================
def load_data(num_of_sample , shufle_data, look_back , method, range_normalize):
    # load the dataset
    dataframe = []
    dataframe = pd.read_csv('data.csv')
    dt = dataframe.values
    if num_of_sample !=0:
        GrundTruth = dt[0:num_of_sample,1]
    else:
        GrundTruth = dt[:,1]
    
    #Normalize
    minimum = np.amin(GrundTruth)
    maximum = np.amax(GrundTruth)
    
    d = range_normalize[0]
    b = range_normalize[1]
    m = (b - d) / (maximum - minimum)
    GrundTruth = (m * (GrundTruth - minimum)) + d

    X, Y = create_series(GrundTruth, look_back, method)
    
    if shufle_data == True:
        shuffle_indices = np.random.permutation(np.arange(len(Y)))
        X = X[shuffle_indices]
        Y = Y[shuffle_indices]
    
    if num_of_sample !=0:
        X = X[0:num_of_sample]
        Y = Y[0:num_of_sample]
    
    return X, Y



# =============================================================================
#  embedding images
# =============================================================================
def create_image_embed(X, model, nTauShiftAmnt, nDimNumOfShifts):
    
    print('Embedding images...')
    embedding = []
    all_Image = []
    
    for i in range(len(X)):
        if i%10 == 0 and len(X) > 100:
            print('image: \t' + str(i+1) + '\t from \t' + str(len(X)))
        adInpAr = X[i]
        Irp1 = MyfGetRcImgM1(adInpAr, nTauShiftAmnt, nDimNumOfShifts)  
        Irp2 = skimage.color.gray2rgb(Irp1)
        all_Image.append(Irp2)
        Irp = Irp2            
        Irp = img_to_array(Irp).astype("float64")
        Irp = transform.resize(Irp, (224, 224))
        Irp *= 1. / 255
        Irp = np.expand_dims(Irp, axis = 0)
        embedding.append(model.predict([Irp,Irp,Irp])[0])
    return all_Image, embedding




# =============================================================================
#  get class to each image
# =============================================================================
def get_label(X, Y, model, num_of_class, train_range_split, method):

    from sklearn.cluster import KMeans
    newY = np.array(Y).reshape(-1,1)
    kmeans = KMeans(n_clusters = num_of_class, random_state = 0).fit(newY)
    classs = kmeans.labels_
    
    set_label = {}
    for i in range(len(Y)):
        if classs[i] in set_label:
            set_label[classs[i]] .append(i)
        else:
            set_label[classs[i]] = []
            set_label[classs[i]] .append(i)
    
    centroids = kmeans.cluster_centers_
    head_img = {}
    for j in range (len(centroids)):
        minim = np.inf
        for i in range(len(Y)):
            if centroids[j] == Y[i]:
                head_img[j]= i
                break
            else:
                dist = (centroids[j] - Y[i])**2
                if dist < minim:
                    minim = dist
                    head_img[j]= i  
    
    all_head = []
    center_image=[]
    for j in head_img:
        all_head.append(head_img[j])
        center_image.append(X[head_img[j]])
    
    img, emb = create_image_embed(center_image, model, nTauShiftAmnt, nDimNumOfShifts)
    
    for i in range(len(img)):
        image=img[i]
        fig = plt.figure()
        plt.imshow(image)
        fig.suptitle('center image for class \t' + str(i+1))
        plt.savefig('Result/' + method + '_class_' + str(i+1))
    
    return classs, set_label, head_img, all_head




# =============================================================================
# calculate accuracy
# =============================================================================
def get_acc(embedding, label, head_img, all_head):
    
    print('Test is started....')
    
    embedding = np.array(embedding)
    shuffle_indices = np.random.permutation(np.arange(len(label)))
    embedding = embedding[shuffle_indices]
    label = label[shuffle_indices]
    
    
#    train_len = int(len(embedding) * train_range_split)
#    x_train = embedding[:train_len]
#    y_train = label[:train_len]
#    x_test = embedding[train_len:]
#    y_test = label[train_len:]
        

    y_pred = []
    y_real = []
    for i in range(len(embedding)):
        if i in all_head:
            continue
        else:
            if i%50 == 0:
                print('predict  ' + str(i) + '   sample from   ' + str(len(embedding)))
            embedding1 = embedding[i,:]
            minim = np.inf
            for j in range(len(head_img)):
                embedding2 = embedding[head_img[j],:]
                distance = sum([(embedding1[idx] - embedding2[idx])**2 for idx in range(len(embedding1))])**(0.5)
                
                if distance < minim:
                    minim = distance
                    k = j
            y_pred.append(label[head_img[k]])
            y_real.append(label[i])
    
    y_pred = np.array(y_pred)
    acc = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_real[i]:
            acc+=1
    acc = acc/len(y_pred)
    print('\n'+ 'Accuracy on test data is: \t' + str(acc))
    return acc, y_pred
    


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    
    nTauShiftAmnt = 4
    nDimNumOfShifts = 3
    num_of_class = 5
    num_of_sample = 200 # 0= all data
    range_normalize = [1,10]
    shufle_data = False
    train_range_split = 0.8
    
    look_back = 50
    method = 'continues' # 'seprate' ,continues
    
    model = deep_rank_model()
    
    X, Y = load_data(num_of_sample, shufle_data, look_back, method, range_normalize)
    
    all_Image, embedding = create_image_embed(X, model, nTauShiftAmnt, nDimNumOfShifts)


#    np.save('Embed5000_seprate.npy', embedding)
#    embedding = np.load('Embed200.npy') #  Embed5000_seprate, Embed200
    
    
    label, set_label, head_img, all_head = get_label(X, Y, model, num_of_class, train_range_split, method)
    
    acc, y_pred = get_acc(embedding, label, head_img, all_head)
    
    
    