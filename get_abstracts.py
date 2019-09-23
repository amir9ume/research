import csv
f= open('metadata.csv', 'r')
content=f.read()
data= content.splitlines()
csvData=[]

#this is getting the paper id ,paper title and abstract : id,title,"asbtract"
for d in data:
	x= d.split('"')
	l=[]
	if len(x)==5:
                z= x[0].split(',')[0]
                l.append(z)
                l.append(x[1])
                l.append(x[3])
                print(l)
                csvData.append(l)


with open('extracted_abstracts.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()



