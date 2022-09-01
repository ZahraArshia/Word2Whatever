import pandas as pd
from openpyxl import load_workbook
#columns: position(temp),value(embedded word)
df = pd.DataFrame({'position':[],
                   'value':[]})
#create excel writer
writer = pd.ExcelWriter('demo.xlsx')
df.to_excel(writer, sheet_name='signal', index=False)
writer.save()
i=0
#open file
writer.book = load_workbook('demo.xlsx')
with open('/content/drive/MyDrive/word2vec.txt','r') as file: 
    # reading each line     
    for line in file: 
   
        # reading each word         
        for word in line.split(): 
             
            #print(word)
            df = pd.DataFrame({'position': [word],
                   'value': [i]})
            i=i+1
            writer = pd.ExcelWriter('demo.xlsx')
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            #read file
            reader = pd.read_excel(r'demo.xlsx')
            #write
            df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)

writer.close()