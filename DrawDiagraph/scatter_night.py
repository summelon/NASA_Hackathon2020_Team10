import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
with open('/file/path') as csvfile:
    spamreader = csv.reader(csvfile)
    x=[]
    x1=['215C','228C','230C','239C','302C']
    x2=[]
    y=[]
    y1=[]
    y2=[]
    c=[]
    e1=[0,0,0,0,0]
    e2=[0,0,0,0,0]
    for row in spamreader:
        for element in row:
            if(element != row[0] and element != row[1] and element !='' and element != '\n'):
                x.append(row[1])
                y.append(float(element))
                if(row[0]=='classification'):
                    c.append('coral')
                    y1.append(float(element))
                else:
                    c.append('palegreen')
                    y2.append(float(element))
                if(row[1]=='215C'):
                        e1[0]+=float(element)
                        x2.append(0)
                elif(row[1]=='215F'):
                        e2[0]+=float(element)
                elif(row[1]=='228C'):
                        e1[1]+=float(element)
                        x2.append(1)
                elif(row[1]=='228F'):
                        e2[1]+=float(element)
                elif(row[1]=='230C'):
                        e1[2]+=float(element)
                        x2.append(2)
                elif(row[1]=='230F'):
                        e2[2]+=float(element)
                elif(row[1]=='239C'):
                        e1[3]+=float(element)
                        x2.append(3)
                elif(row[1]=='239F'):
                        e2[3]+=float(element)
                elif(row[1]=='302C'):
                        e1[4]+=float(element)
                        x2.append(4)
                elif(row[1]=='302F'):
                        e2[4]+=float(element)
                else:
                    pass

ticks=['Rare Case 1','Rare Case 2','Rare Case 3','Rare Case 4','Rare Case 5']
for i in range(0,5):
    e1[i]=round(e1[i]/5,2)
    e2[i]=round(e2[i]/5,2)
fig, ax=plt.subplots()
fig=plt.figure()
ax=fig.add_subplot()
ax=plt.gca()
ax.set_facecolor((0.5, 0.5, 0.5))
plt.title('Night View (F1-score)')
plt.grid(b=True, axis='y',alpha=0.3,linewidth=1.2)

plt.bar(np.arange(len(x1))*2,np.array(e1),width=0.95,alpha=0.4,color='coral')
plt.bar(np.arange(len(x1))*2+0.95,np.array(e2),width=0.95,alpha=0.4,color='palegreen')
plt.scatter(np.array(x2)*2,np.array(y1),color='tomato')
plt.scatter(np.array(x2)*2+0.95,np.array(y2),color='springgreen')
l1=plt.plot([],c='tomato',label='Naïve Sample',marker='o',linestyle='')
l2=plt.plot([],c='limegreen',label='Few-shot Sample',marker='o',linestyle='')
l3=plt.plot([],c='tomato',label='Naïve Expectation',marker='s',linestyle='')
l4=plt.plot([],c='limegreen',label='Few-shot Expectation',marker='s',linestyle='')

plt.legend()
plt.xticks(np.arange(0.5,12.5,step=2),ticks)
plt.xlim(-1,len(ticks)*2)
plt.ylim(-0.01,1.19)
plt.tight_layout()
plt.savefig('/image/path')