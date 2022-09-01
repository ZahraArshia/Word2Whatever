def create_series(dataset, method): 
    X = []
    length = len(dataset)
    
    if method == 'byword':
        for i in range(0, length-300, 1):
             sequence = dataset[i:i+300]
             X.append(sequence)
    else:
        for i in range(0, length, 300):
             sequence = dataset[i:i+299]
             X.append(sequence)
      
    return X
	