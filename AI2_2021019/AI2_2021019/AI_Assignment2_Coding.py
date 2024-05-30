import heapq
import csv
import random
table=[]
cities=[]
with open("Road_Distance.csv", mode="r")as file:
    fi=csv.reader(file)
    for line in fi:
        table.append(line)
cities+=table[0][1:]
for i in range(1,len(table)):
    if table[i][0] not in cities:
        cities.append(table[i][0])
graph={}
for city in cities:
    graph[city]=[]
for i in range(1,len(table)):
    for j in range(1,len(table[0])):
        if table[i][j]!="-":
            graph[table[i][0]].append([table[0][j],table[i][j]])
            graph[table[0][j]].append([table[i][0], table[i][j]])
def dist(graph, start, goal):
    pqueue=[(0,0,start,[])]
    if start==goal:
        return 0
    while pqueue:
        useless,cost,curr,path=heapq.heappop(pqueue)
        if curr==goal:
            return cost
        for adj,ecost in graph[curr]:
            necos=cost + int(ecost)
            priority=necos
            heapq.heappush(pqueue,(priority,necos,adj,path+[curr]))
    return None
def admissible(graph,city, goal):
    if city==goal:
        return 0
    return dist(graph,city,goal)-random.randint(7,10)
def inadmissible(graph,city,goal):
    if city==goal:
        return 0
    return 100000-dist(graph,city,goal)
def astar(graph,start,goal,hue):
    pqueue = [(0+hue(graph, start, goal), 0, start, [])]
    while pqueue:
        useless,cost,curr,path=heapq.heappop(pqueue)
        if curr==goal:
            return path+[curr]
        for adj, ecost in graph[curr]:
            necos = cost + int(ecost)
            npat=path+[curr]
            priority=necos+hue(graph,adj,goal)
            heapq.heappush(pqueue,(priority,necos,adj,npat))
    return None

print(1)
def func(a,b,c): # trivial heuristic
    return 0
# print(astar(graph,'Hubli','Chandigarh',inadmissible))
print("Road Distance Program:-")
while(1!=0):
    x=input("Source:- ")
    y=input("Destination:-")
    print("Which of the following algorithms would you like to use to find the shortest distance?:-")
    if x not in cities or y not in cities:
        print("Incorrect Submission")
        break
    print("1. Uniform Cost Search")
    print("2. A*")
    meth=int(input(":- "))
    if meth==1:
        print(astar(graph,x,y,func))
        print(dist(graph,x,y))
    else:
        print("Which heuristic would you prefer:-")
        print("1. Admissible")
        print("2. Inadmissible")
        choice=int(input())
        if choice==1:
            print(astar(graph,x,y,admissible))
            print(dist(graph,x,y))
        else:
            print(astar(graph,x,y,inadmissible))