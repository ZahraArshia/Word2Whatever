def create_label(dataset, method): #label=position(temporary)
    X = []
    Y = []
    length = len(dataset)
    
    if method == 'byword':
        for i in range(0, length-300, 1):
             sequence = dataset[i:i+300]
             label = dataset[i]
             X.append(sequence)
             Y.append(label)
    else:
        for i in range(0, length, 300):
             sequence = dataset[i:i+299]
             label = dataset[i]
             X.append(sequence)
             Y.append(label)
      
    return X,Y
	