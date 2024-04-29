import traceback
import copy

from utils import *

class NodeType:
    def __init__(self,nodeTypeName):
        self.nodeTypeName=nodeTypeName
class Node:
    def __init__(self):
        self.id = 0
        self.strEmb=''
        self.lstEdges=[]
    def __init__(self, nodeType, id,lstEdges,strEmb):
        self.id = id
        self.strEmb = strEmb
        self.lstEdges=lstEdges

class EdgeType:
    def __init__(self,edgeName,sourceType,targetType):
        self.edgeName=edgeName
        self.sourceType=sourceType
        self.targetType=targetType


class Edge:
    def __init__(self,sourceNode,targetNode):
        self.sourceNode = sourceNode
        self.targetNode=targetNode

class GraphGenV2:

    def __init__(self,fopGraph,lstScaleSize):
        self.lstScaleSize=lstScaleSize
        self.fopGraph=fopGraph
        self.dictContentNodes={}
        self.dictContentEdges={}
        self.lstNodeTypes=[]
        self.lstEdgeTypes=[]

    def analyzePerfCounter(self,X,y):
        lenFeatures=len(X[0])
        dictPf={}
        for i in range(0,lenFeatures):
            listFeatI=[X[j][i] for j in range(0,len(X))]
            listFeatI=sorted(listFeatI)
            # setFeatI=list(set(listFeatI))
            minFeat=min(listFeatI)
            maxFeat=max(listFeatI)
            numOfSplits=self.lstScaleSize[i]-1
            offsetSplit=(maxFeat-minFeat)//(numOfSplits+1)
            lstScales=[]
            for j in range(1,numOfSplits+1):
                valAtPos=minFeat+offsetSplit*j
                lstScales.append(valAtPos)
            dictPf[i]=lstScales
        return dictPf


    def createGraphStructure(self,dictPf,X,y):
        dictAllNodes={}
        dictAllEdges={}
        nodeTypeJobId=NodeType('JobId')
        lstNodeTypePf = [NodeType('PerfCountN{}'.format(i)) for i in range(0,len(X[0]))]
        self.lstNodeTypes=[nodeTypeJobId]+lstNodeTypePf
        dictNodeTypes = {}
        for item in self.lstNodeTypes:
            dictNodeTypes[item.nodeTypeName] = item
        self.lstEdgeTypes=[]
        for i in range(1,len(self.lstNodeTypes)):
            edgeJobPf=EdgeType('job-pf{}'.format(i),self.lstNodeTypes[0],self.lstNodeTypes[i])
            edgePfJob = EdgeType('pf{}-job'.format(i),self.lstNodeTypes[i],self.lstNodeTypes[0])
            self.lstEdgeTypes.append(edgeJobPf)
            self.lstEdgeTypes.append(edgePfJob)

        dictEdgeTypes={}
        for item in self.lstEdgeTypes:
            dictEdgeTypes[item.edgeName]=item
        dictGraphStructure = {}
        dictGraphStructure['dictNodeTypes']=dictNodeTypes
        dictGraphStructure['dictEdgeTypes'] = dictEdgeTypes

        indexNodes=0

        lstYamlTrain = ['dataset_name: {}\nedge_data:\n'.format(self.fopGraph)]
        # lstYamlTest=['dataset_name: {}\nedge_data:\n'.format(fopCsvGNNTest)]
        for i in range(0,len(self.lstEdgeTypes)):
            edgeItem=self.lstEdgeTypes[i]
            strEdge = '- file_name: edges_{}.csv\n  etype: [{}, {}, {}]'.format(i, edgeItem.sourceType.nodeTypeName, edgeItem.edgeName, edgeItem.targetType.nodeTypeName)
            self.dictContentEdges[edgeItem.edgeName]='edges_{}.csv'.format(i)
            f1=open(self.fopGraph+self.dictContentEdges[edgeItem.edgeName],'w')
            f1.write('graph_id,src_id,dst_id\n')
            f1.close()
            lstYamlTrain.append(strEdge)

        lstYamlTrain.append('node_data:')
        # lstYamlTest.append('node_data:')

        for i in range(0, len(self.lstNodeTypes)):
            nodeItem=self.lstNodeTypes[i]
            strNode = '- file_name: nodes_{}.csv\n  ntype: {}'.format(i, nodeItem.nodeTypeName)
            self.dictContentNodes[nodeItem.nodeTypeName] = 'nodes_{}.csv'.format(i)
            f1 = open(self.fopGraph + self.dictContentNodes[nodeItem.nodeTypeName], 'w')
            f1.write('graph_id,node_id,feat\n')
            f1.close()
            lstYamlTrain.append(strNode)

        strGraphInfo = 'graph_data:\n  file_name: graphs.csv'
        lstYamlTrain.append(strGraphInfo)
        # lstYamlTest.append(strGraphInfo)

        f1 = open(self.fopGraph + 'meta.yaml', 'w')
        f1.write('\n'.join(lstYamlTrain))
        f1.close()
        f1=open(self.fopGraph+'graphs.csv','w')
        f1.write('graph_id,label\n')
        f1.close()

        cacheSize=5000
        dictWriteToFileTotal={}
        for key in self.dictContentNodes.keys():
            dictWriteToFileTotal[self.dictContentNodes[key]]=[]
        for key in self.dictContentEdges.keys():
            dictWriteToFileTotal[self.dictContentEdges[key]]=[]
        dictWriteToFileTotal['graphs.csv']=[]

        for i in range(0,len(X)):
            isSuccess=False
            try:
                itemI = X[i]
                yI = y[i]
                dictWriteToFileSingle = self.generateGraphEntityFromArrayList(dictGraphStructure,i, itemI, yI)
                isSuccess=True
                if isSuccess:
                    for key in dictWriteToFileSingle.keys():
                        lstVal2=copy.copy(dictWriteToFileSingle[key])
                        dictWriteToFileTotal[key]+=(lstVal2)


            except Exception as e:
                traceback.print_exc()

            if (i+1)%cacheSize==0 or len(X)==(i+1):
                for key in dictWriteToFileTotal.keys():
                    lstVal=dictWriteToFileTotal[key]
                    if len(lstVal)>0:
                        f1=open(self.fopGraph+key,'a')
                        f1.write('\n'.join(lstVal)+'\n')
                        f1.close()

                dictWriteToFileTotal = {}
                for key in self.dictContentNodes.keys():
                    dictWriteToFileTotal[self.dictContentNodes[key]] = []
                for key in self.dictContentEdges.keys():
                    dictWriteToFileTotal[self.dictContentEdges[key]] = []
                dictWriteToFileTotal['graphs.csv'] = []
                print('end {}'.format(i))
    def getScaleScore(self,inputValue,lstScales):
        finalScale=len(lstScales)
        for i in range(0,len(lstScales)):
            if inputValue<=lstScales[i]:
                finalScale=i+1
                break
        return finalScale


    def generateGraphEntityFromArrayList(self,dictGraphStructure,id,arrFeatItem,label):
        dictInfoToGraphs={}
        # try:
        strLine = '{},{},"{}"'.format(id, 0, ','.join(map(str, arrFeatItem)))
        dictInfoToGraphs[self.dictContentNodes[self.lstNodeTypes[0].nodeTypeName]]=[strLine]
        for i in range(0,len(arrFeatItem)):
            valInput=arrFeatItem[i]
            lstValInput=[valInput for j in range(0,5)]
            strValInput='{},{},"{}"'.format(id, (i+1), ','.join(map(str, lstValInput)))
            dictInfoToGraphs['nodes_{}.csv'.format(i+1)]=[strValInput]
            strGraphLine1='{},{},{}'.format(id,0,(i+1))
            strGraphLine2 = '{},{},{}'.format(id, (i + 1), 0)
            dictInfoToGraphs[self.dictContentEdges['job-pf{}'.format(i+1)]]=[strGraphLine1]
            dictInfoToGraphs[self.dictContentEdges['pf{}-job'.format(i + 1)]] = [strGraphLine2]
        dictInfoToGraphs['graphs.csv']=['{},{}'.format(id,label)]
        # except Exception as e:
        #     traceback.print_exc()
        return dictInfoToGraphs