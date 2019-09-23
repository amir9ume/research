import csv
f= open('metadata.csv', 'r')
content=f.read()
data= content.splitlines()
csvData=[]

#this was just getting the author names and paper title
for d in data:
	x= d.split('"')
	l=[]
	if len(x)==5:
		z= x[0].split(',')[0]
		l.append(z)
		l.append(x[1])	
		print(l)
	csvData.append(l)


with open('test.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()

		


